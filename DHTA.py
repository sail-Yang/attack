import torch
from omegaconf import OmegaConf
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_attack_hashing_logger
from utils.util import *

torch.multiprocessing.set_sharing_strategy("file_system")

def config_dhta(args):
  '''
  @return
    model: HashModel
    database_code_path: str
    t_model: attacked HashModel
    t_database_code_path: str
    target_label_path: str
    test_code_path: str
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
  return model, database_code_path, t_model, t_database_code_path, target_label_path, test_code_path

def get_labels_and_codes(args, train_loader, test_loader, database_loader, num_train, num_test, num_database):
  model, database_code_path, t_model, t_database_code_path, target_label_path, test_code_path = config_dhta(args)
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
  
  


if __name__ == "__main__":
  conf_root = "./configs/DHTA.yaml"
  args = load_config(conf_root)
  seed_setting(args.seed)
  logger = create_attack_hashing_logger(args)
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  get_labels_and_codes(args, train_loader, test_loader, database_loader, num_train, num_test, num_database)