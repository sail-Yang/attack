import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_attack_hashing_logger
from utils.util import *
from utils.validate import CalcTopMap
import collections
import pandas as pd

torch.multiprocessing.set_sharing_strategy("file_system")

def config_dhta(args):
  '''
  @return
    model: HashModel
    t_model: attacked HashModel
    database_code_path: str
    t_database_code_path: str
    target_label_path: str
    test_code_path: str
    t_bit: int
  '''
  if args.n_t == 1:
    args.attack_method = "p2p"
  hashModel = HashModel(args)
  hashModel.load_model()
  model = hashModel.model.cuda()
  database_code_path = os.path.join(args.save_path, args.attack_method, "database_code_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  if args.transfer:
    t_hash_model = args.trans_config.t_hash_model
    t_bit = args.trans_config.t_bit
    t_backbone = args.trans_config.t_backbone
    t_model_path = os.path.join(args.hash_save_path, t_hash_model, '{}_{}_{}_{}.pt'.format(t_hash_model, t_backbone, t_bit))
    t_model = HashModel.load_t_model(t_model_path)
  else:
    t_hash_model = args.hash_model
    t_bit = args.num_bits
    t_backbone = args.backbone
    t_model = model
  t_database_code_path = os.path.join(args.save_path, args.attack_method, "database_code_{}_{}_{}_{}.txt".format(args.dataset, t_hash_model, t_backbone, t_bit))
  target_label_path = os.path.join(args.save_path, args.attack_method, "target_label_{}.txt".format(args.dataset))
  test_code_path = os.path.join(args.save_path, args.attack_method, "test_code_{}_{}_{}.txt".format(args.dataset, args.attack_method, t_bit))
  return model, t_model, database_code_path, t_database_code_path, target_label_path, test_code_path, t_bit

def get_target_labels(database_txt_path, num_test, n_t):
  '''
    @params:
      database_txt_path: str
      num_test: int
      n_t: int
    @return:
      target_labels: np.array
  '''
  database_labels_str = get_labels_str(database_txt_path)
  
  ### 从database labels中统计标签出现次数并筛选出次数大于n_t的标签
  # 生成字典，键为标签，值为出现次数
  candidate_labels_count = collections.Counter(database_labels_str)
  # 字典转换为DataFrame，第一列为标签，第二列为统计的数
  candidate_labels_count = pd.DataFrame.from_dict(candidate_labels_count, orient='index').reset_index()
  # 筛选出次数大于n_t的标签
  candidate_labels = candidate_labels_count[candidate_labels_count[0] > n_t]['index']
  candidate_labels = np.array(candidate_labels, dtype=str)
  
  target_labels = []
  for i in range(num_test):
    target_label_str = np.random.choice(candidate_labels)
    target_label = list(target_label_str)
    target_label = np.array(target_label, dtype=np.int64)
    target_labels.append(target_label)
  
  target_labels = np.array(target_labels, dtype=np.int64)
  return target_labels

def get_labels_and_codes(args, model, t_model, database_code_path, t_database_code_path, target_label_path, database_loader, num_test, num_database):
  '''
  @params:
    args: argparse.Namespace
    model/t_model: HashModel
    database_code_path/t_database_code_path/target_label_path: str
    database_loader: torch.utils.data.DataLoader
    num_test/num_database: int
  @return:
    database_hash: np.array
    t_database_hash: np.array
    target_labels_str: list
  '''
  # load database code
  if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=np.float32)
  else:
    database_hash = generateCode(model, database_loader, num_database, args.num_bits)
    np.savetxt(database_code_path, database_hash, fmt="%d")
  
  # load trasfer database code
  if os.path.exists(t_database_code_path):
    t_database_hash = np.loadtxt(t_database_code_path, dtype=np.float32)
  else:
    t_database_hash = generateCode(t_model, database_loader, num_database, args.trans_config.t_bit)
    np.savetxt(t_database_code_path, t_database_hash, fmt="%d")
  logger.info("database hash codes prepared!")
  
  # load target label
  if os.path.exists(target_label_path):
    target_labels = np.loadtxt(target_label_path, dtype=np.int64)
  else:
    database_txt_path = os.path.join(args.txt_path, "database_label.txt")
    target_labels = get_target_labels(database_txt_path, num_test, args.n_t)
    np.savetxt(target_label_path, target_labels, fmt="%d")
  
  return database_hash, t_database_hash, target_labels

def vote_anchor_code(hash_codes):
  '''
    vote for generation of anchor code
    @params:
      hash_codes: torch.Tensor
    @return:
      anchor_code: torch.Tensor
  '''
  return torch.sign(torch.sum(hash_codes, dim=0))

def target_adv_loss(noisy_output, target_hash):
  loss = -torch.mean(noisy_output * target_hash)
  return loss

def target_hash_adv(model, query, target_hash, epsilon, alpha=1, iteration=2000, randomize = False):
  '''
    iterate attack
  '''
  delta = torch.zeros_like(query, requires_grad=True).cuda()
  if randomize:
    delta.uniform_(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
  for i in range(iteration):
    # noisy_output = model(query + delta, factor)
    noisy_output = model(query + delta)
    loss = target_adv_loss(noisy_output, target_hash)
    loss.backward()
    
    delta.data = delta - alpha * delta.grad.detach()
    delta.data = delta.data.clamp(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.grad.zero_()
  
  return query + delta.detach()
    
    

if __name__ == "__main__":
  conf_root = "./configs/DHTA.yaml"
  args = load_config(conf_root)
  seed_setting(args.seed)
  logger = create_attack_hashing_logger(args)
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  model, t_model, database_code_path, t_database_code_path, target_label_path, test_code_path, t_bit = config_dhta(args)
  database_hash, t_database_hash, target_labels = get_labels_and_codes(args, model, t_model, database_code_path, t_database_code_path, target_label_path, database_loader, num_test, num_database)
  target_labels_str = [''.join(label) for label in target_labels.astype(str)]
  
  # 获取database labels
  database_txt_path = os.path.join(args.txt_path, "database_label.txt")
  database_labels_str = get_labels_str(database_txt_path)
  qB = np.zeros([num_test, t_bit], dtype=np.float32)
  query_anchor_codes = np.zeros((num_test, args.num_bits), dtype=np.float32)
  perceptibility = 0
  
  for it, data in enumerate(test_loader):
    image, label, ind = data
    image = image.cuda()
    # 这里可能dataloader最后一组不够args.batch_size，所以用batch_size_
    batch_size_ = ind.size(0)
    anchor_codes = torch.zeros((batch_size_, args.num_bits), dtype=torch.float)
    for i in range(batch_size_):
      target_label_str = target_labels_str[ind[i]]
      anchor_indexes = np.where(database_labels_str == target_label_str)
      anchor_indexes = np.random.choice(anchor_indexes[0], size=args.n_t)
      
      anchor_code = vote_anchor_code(
        torch.from_numpy(database_hash[anchor_indexes]))
      anchor_code = anchor_code.view(1, args.num_bits)
      anchor_codes[i, :] = anchor_code
    query_anchor_codes[it*args.batch_size:it*args.batch_size+batch_size_] = anchor_codes.numpy()
    query_adv = target_hash_adv(model, image, anchor_codes.cuda(), epsilon=args.epsilon, iteration=args.iteration)
    if args.transfer:
      query_code = generateHash(t_model, query_adv)
    else:
      query_code = generateHash(model, query_adv)
    # 生成索引数组
    u_ind = np.linspace(it * args.batch_size, np.min((num_test, (it+1) * args.batch_size)) - 1, batch_size_, dtype=int)
    qB[u_ind, :] = query_code
    perceptibility += F.mse_loss(image, query_adv).data * batch_size_

np.savetxt(test_code_path, qB, fmt="%d")
database_labels_int = get_labels_int(database_txt_path)
test_txt_path = os.path.join(args.txt_path, "test_label.txt")
test_labels_int = get_labels_int(test_txt_path)

logger.info("perceptibility: {:.7f}".format(torch.sqrt(perceptibility/num_test)))
anchor_map = CalcTopMap(t_database_hash, query_anchor_codes, database_labels_int, target_labels, topk=args.topK)
logger.info("anchor codes t-MAP[retrieval database]: {:.7f}".format(anchor_map))
t_map = CalcTopMap(t_database_hash, qB, database_labels_int, target_labels, topk=args.topK)
logger.info("t-MAP[retrieval database]: {:.7f}".format(t_map))
map = CalcTopMap(t_database_hash, qB, database_labels_int, test_labels_int, topk=args.topK)
logger.info("MAP[retrieval database]: {:.7f}".format(map))
