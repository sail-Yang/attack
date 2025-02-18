import logging
import time
import os

def create_hashing_logger(args):
  logger_file_path = args.log_path
  hash_model = args.hash_model
  backbone = args.backbone
  dataset = args.dataset
  num_bits = args.num_bits
  log_time = time.strftime('%Y-%m-%d-%H-%M')
  log_dir_path = os.path.join(logger_file_path, hash_model)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
  log_name = f"{backbone}_{dataset}_{num_bits}_{log_time}.log"
  log_path = os.path.join(logger_file_path, hash_model, log_name)
  
  logger = logging.getLogger() 
  logger.setLevel(logging.INFO) 
  
  file_handler = logging.FileHandler(log_path)
  console_handler = logging.StreamHandler()
  
  # format
  formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s"
  )
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  
  return logger

def create_attack_hashing_logger(args):
  logger_file_path = args.log_path
  attack_method = args.attack_method
  backbone = args.backbone
  dataset = args.dataset
  num_bits = args.num_bits
  hash_model = args.hash_model
  log_time = time.strftime('%Y-%m-%d-%H-%M')
  log_dir_path = os.path.join(logger_file_path, attack_method, hash_model)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
  if args.transfer:
    log_name = f"transfer_{backbone}_{dataset}_{num_bits}_{log_time}.log"
  else:
    log_name = f"{backbone}_{dataset}_{num_bits}_{log_time}.log"
  log_path = os.path.join(logger_file_path, attack_method, hash_model, log_name)
  
  logger = logging.getLogger() 
  logger.setLevel(logging.INFO) 
  
  file_handler = logging.FileHandler(log_path)
  console_handler = logging.StreamHandler()
  
  # format
  formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s"
  )
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)
  
  return logger