import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_attack_hashing_logger
from utils.util import *
from utils.validate import CalcTopMap,CalcMap
from models.prototypeNet import PrototypeNet
import collections
import pandas as pd

torch.multiprocessing.set_sharing_strategy("file_system")

def config_tha(args):
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
  hashModel = HashModel(args, args.hash_model, args.backbone, args.num_bits)
  hashModel.load_model()
  model = hashModel.model.cuda()
  database_code_path = os.path.join(args.save_path, args.attack_method, "database_code_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  if args.transfer:
    t_hash_model = args.trans_config.t_hash_model
    t_bit = args.trans_config.t_bit
    t_backbone = args.trans_config.t_backbone
    logger.info("target model: {} {} {}".format(t_hash_model, t_backbone, t_bit))
    
    t_model = HashModel(args, t_hash_model, t_backbone, t_bit)
    t_model.load_model()
    t_model = t_model.model.cuda()
  else:
    t_hash_model = args.hash_model
    t_bit = args.num_bits
    t_backbone = args.backbone
    t_model = model
  t_database_code_path = os.path.join(args.save_path, args.attack_method, "database_code_{}_{}_{}_{}.txt".format(args.dataset, t_hash_model, t_backbone, t_bit))
  target_label_path = os.path.join(args.save_path, args.attack_method, "target_label_{}.txt".format(args.dataset))
  test_code_path = os.path.join(args.save_path, args.attack_method, "test_code_{}_{}_{}.txt".format(args.dataset, args.attack_method, t_bit))
  return model, t_model, database_code_path, t_database_code_path, target_label_path, test_code_path, t_bit

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
  
  return database_hash, t_database_hash

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
    
    # delta.data = delta - alpha / 255 * torch.sign(delta.grad.detach())
    delta.data = delta - alpha * delta.grad.detach()
    delta.data = delta.data.clamp(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.grad.zero_()
  return query + delta.detach()
    
def train_pnet(args, model, train_loader, train_labels, num_train):
  pnet_path = os.path.join(args.save_path, args.attack_method, "PrototypeNet_{}_{}_{}_{}.txt".format(args.dataset,args.hash_model, args.backbone, args.num_bits))
  pnet = PrototypeNet(args.num_bits, args.n_class).cuda()
  if os.path.exists(pnet_path):
    pnet.load_model(pnet_path)
  else:
    optimizer_l = torch.optim.Adam(pnet.parameters(), lr=args.lr, betas=(0.5, 0.999))        
    epochs = 100
    steps = 300
    batch_size = args.batch_size
    lr_steps = epochs * steps
    # 调整学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    circle_loss = CircleLoss(m=0, gamma=1)
    
    # hash codes of training set
    B = generateCode(model, train_loader, num_train, args.num_bits)
    B = torch.from_numpy(B).cuda()
    
    for epoch in range(epochs):
      for i in range(steps):
        select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
        batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        
        optimizer_l.zero_grad()
        
        _,target_hash_l,_ = pnet(batch_target_label)
        sp, sn = similarity(target_hash_l, B, batch_target_label, train_labels.cuda(), args.num_bits)
        logloss = circle_loss(sp, sn) / (args.batch_size)
        regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
        loss = logloss + regterm
        
        loss.backward()
        optimizer_l.step()
        if i % args.checkpoint == 0:
          logger.info("epoch: {:2d}, step: {:3d}, lr: {:5f}, logloss:{:.5f}, regterm: {:.5f}".format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm))
        scheduler.step()
    
    torch.save(pnet.state_dict(), pnet_path)
    pnet.eval()
    
  return pnet
    
    
class CircleLoss(nn.Module):
  def __init__(self, m: float, gamma: float) -> None:
    super(CircleLoss, self).__init__()
    self.m = m
    self.gamma = gamma
    self.soft_plus = nn.Softplus()

  def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
    ap = torch.clamp_min(- sp.detach() + 2, min=0.)
    an = torch.clamp_min(sn.detach() + 2, min=0.)

    logit_p = - ap * sp * self.gamma
    logit_n = an * sn * self.gamma
    loss = torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
    return loss

def similarity(batch_feature, features, batch_label, labels, bit):
  # 计算相似度矩阵
  similarity_matrix = batch_feature @ features.transpose(1, 0)
  similarity_matrix = similarity_matrix / bit
  
  # 标签矩阵
  label_matrix = (batch_label.mm(labels.t()) > 0)
  
  # 提取上三角，得到正样本矩阵
  positive_matrix = label_matrix.triu(diagonal=1)
  # 逻辑非后取上三角得到负样本矩阵
  negative_matrix = label_matrix.logical_not().triu(diagonal=1)
  
  # 展平向量
  similarity_matrix = similarity_matrix.view(-1)
  positive_matrix = positive_matrix.view(-1) 
  negative_matrix = negative_matrix.view(-1)
  
  # 获取相似度值
  return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

if __name__ == "__main__":
  conf_root = "./configs/tha.yaml"
  args = load_config(conf_root)
  seed_setting(args.seed)
  logger = create_attack_hashing_logger(args)
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  train_txt_path = os.path.join(args.txt_path, "train_label.txt")
  database_txt_path = os.path.join(args.txt_path, "database_label.txt")
  test_txt_path = os.path.join(args.txt_path, "test_label.txt")
  train_labels = torch.from_numpy(get_labels_int(train_txt_path)).float()
  database_labels = torch.from_numpy(get_labels_int(database_txt_path)).float()
  test_labels = torch.from_numpy(get_labels_int(test_txt_path)).float()
  target_labels = database_labels.unique(dim=0)
  
  
  model, t_model, database_code_path, t_database_code_path, target_label_path, test_code_path, t_bit = config_tha(args)
  database_hash, t_database_hash = get_labels_and_codes(args, model, t_model, database_code_path, t_database_code_path, target_label_path, database_loader, num_test, num_database)
  # 获取train labels
  train_txt_path = os.path.join(args.txt_path, "train_label.txt")
  train_labels = torch.from_numpy(get_labels_int(train_txt_path)).float()
  
  qB = np.zeros([num_test, t_bit], dtype=np.float32)
  query_prototype_codes = np.zeros((num_test, args.num_bits), dtype=np.float32)
  perceptibility = 0
  
  pnet = train_pnet(args, model, train_loader, train_labels, num_train)
  
  ## 获取所有target labels
  if os.path.exists(target_label_path):
    targeted_labels = np.loadtxt(target_label_path, dtype=np.int64)
  else:
    targeted_labels = np.zeros([num_test, args.n_class])
    for data in test_loader:
      _, label, index = data
      batch_size_ = index.size(0)
      select_index = np.random.choice(range(target_labels.size(0)), size=batch_size_)
      batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
      targeted_labels[index.numpy(), :] = batch_target_label.cpu().data.numpy()
    
    np.savetxt(target_label_path, targeted_labels, fmt="%d")
    
  l0_norm_mean = 0
  for it, data in enumerate(test_loader):
    image, label, ind = data
    batch_size_ = ind.size(0)
    image = image.cuda()
    batch_target_label = targeted_labels[ind.numpy(), :]
    batch_target_label = torch.from_numpy(batch_target_label).float().cuda()
    
    _,batch_prototype_codes,_ = pnet(batch_target_label)
    prototype_codes = torch.sign(batch_prototype_codes)
    query_prototype_codes[ind.numpy(), :] = prototype_codes.cpu().data.numpy()
    
    query_adv = target_hash_adv(model, image, prototype_codes.cuda(), epsilon=args.epsilon, iteration=args.iteration)
    if args.transfer:
      query_code = generateHash(t_model, query_adv)
    else:
      query_code = generateHash(model, query_adv)
    
    # calculate l-0 norm
    diff = torch.ne(image, query_adv)
    l0_norm = torch.sum(diff).item()
    total_pixels = image.numel()
    l0_norm_mean += l0_norm / total_pixels * batch_size_
    
    # 采样图像
    if it % args.sample_checkpoint == 0:
      dir_path = os.path.join(args.save_path, args.attack_method, "sample")
      sample_img(image, dir_path, "{}_ori".format(it))
      sample_img(query_adv, dir_path, "{}_adv".format(it))
      
    # 生成索引数组
    u_ind = np.linspace(it * args.batch_size, np.min((num_test, (it+1) * args.batch_size)) - 1, batch_size_, dtype=int)
    qB[u_ind, :] = query_code
    perceptibility += F.mse_loss(image, query_adv).data * batch_size_
    
    

np.savetxt(test_code_path, qB, fmt="%d")
database_labels_int = get_labels_int(database_txt_path)
test_txt_path = os.path.join(args.txt_path, "test_label.txt")
test_labels_int = get_labels_int(test_txt_path)

logger.info("perceptibility: {:.7f}".format(torch.sqrt(perceptibility/num_test)))
logger.info("L0 norm: {:.7f}".format(l0_norm_mean / num_test))
p_map = CalcMap(t_database_hash, query_prototype_codes, database_labels_int, targeted_labels)
logger.info("prototype codes t-MAP[retrieval database]: {:.7f}".format(p_map))
t_map = CalcMap(t_database_hash, qB, database_labels_int, targeted_labels)
logger.info("t-MAP of adv[retrieval database]: {:.7f}".format(t_map))
map = CalcTopMap(t_database_hash, qB, database_labels_int, test_labels_int, topk=args.topK)
logger.info("MAP of adv[retrieval database]: {:.7f}".format(map))
