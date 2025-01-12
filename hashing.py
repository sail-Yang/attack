import torch
from omegaconf import OmegaConf
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_hashing_logger

torch.multiprocessing.set_sharing_strategy("file_system")

conf_root = "./configs/hash.yaml"
args = load_config(conf_root)
logger = create_hashing_logger(args)

train_loader, test_loader, database_loader = get_data(args)
model = HashModel(args)

if __name__ == "__main__":
  if args.train:
    logger.info(f"{args.hash_model}, {args.backbone}, {args.dataset}, {args.num_bits} bits, training...")
    model.train(train_loader, test_loader, database_loader)
  
  if args.test:
    logger.info(f"{args.hash_model}, {args.backbone}, {args.dataset}, {args.num_bits} bits, testing...")
    model.test_model(test_loader, database_loader)