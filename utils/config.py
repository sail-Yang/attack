from omegaconf import OmegaConf
import torch
import os
def load_config(conf_root_path):
  """
    Load configuration.
  """
  args = OmegaConf.load(conf_root_path)
  if args.gpu is None:
    device = torch.device('cpu')
  else:
    device = torch.device("cuda:%d" % args.gpu)
  torch.cuda.set_device(device)
  # config dataset
  args = config_dataset(args)
  return args

def config_dataset(args):
  '''
  设置数据集参数
  '''
  if args.dataset == "cifar10":
    args.n_class = 10
  elif args.dataset in ["NUS-WIDE","nuswide"] :
    args.dataset = "NUS-WIDE"
    args.n_class = 21
  elif args.dataset in ["coco","MS-COCO"]:
    args.dataset = "MS-COCO"
    args.n_class = 80
  elif args.dataset == "imagenet":
    args.n_class = 100
  args.data_path = os.path.join(args.data_path,args.dataset)
  args.txt_path = os.path.join(args.txt_path,args.dataset)
  return args