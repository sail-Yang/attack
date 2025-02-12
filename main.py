import argparse
from utils.data_provider import *
from models.target_attack_gan import *
from omegaconf import OmegaConf
from utils.config import load_config
from utils.data_provider_copy import *
from utils.log import create_attack_hashing_logger


conf_root = "./configs/pros_gan.yaml"
args = load_config(conf_root)
seed_setting(args.seed)
logger = create_attack_hashing_logger(args)
  
dset_database = HashingDataset('/data2/fyang/data/' + args.dataset, "database_img.txt", "database_label.txt")
dset_train = HashingDataset('/data2/fyang/data/' + args.dataset, "train_img.txt", "train_label.txt")
dset_test = HashingDataset('/data2/fyang/data/' + args.dataset, "test_img.txt", "test_label.txt")
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)
database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=args.batch_size//2, shuffle=True, num_workers=4)

database_labels = load_label("database_label.txt", '/data2/fyang/data/' + args.dataset)
train_labels = load_label("train_label.txt", '/data2/fyang/data/' + args.dataset)
test_labels = load_label("test_label.txt", '/data2/fyang/data/' + args.dataset)
target_labels = database_labels.unique(dim=0)


model = TargetAttackGAN(args=args)
if args.train:
    model.train(train_loader, target_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test)


if args.test:
    model.load_model()
    model.test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
