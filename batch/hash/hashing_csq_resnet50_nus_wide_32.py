import torch
from omegaconf import OmegaConf
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_hashing_logger
from utils.util import *

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
  conf_root = "./configs/hash.yaml"
  args = load_config(conf_root)
  # batch run hashing_dpsh_resnet50_nus_wide_32.py
  args.num_bits = 32
  args.hash_model = "CSQ"
  args.backbone = "ResNet50"
  
  seed_setting(args.seed)
  logger = create_hashing_logger(args)
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  args.dpsh_params.num_train = num_train
  model = HashModel(args)
  if args.train:
    logger.info(f"{args.hash_model}, {args.backbone}, {args.dataset}, {args.num_bits} bits, training...")
    model.train(train_loader, test_loader, database_loader)
  
  if args.test:
    logger.info(f"{args.hash_model}, {args.backbone}, {args.dataset}, {args.num_bits} bits, testing...")
    model.load_model()
    model.test_model(test_loader, database_loader)