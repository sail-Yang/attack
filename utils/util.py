import torch
import os
import random
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

def seed_setting(seed=2021):
  """
  固定随机种子，使得每次训练结果相同，方便对比模型效果
  """
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # torch.backends.cudnn.benchmark = False # False make training process too slow!
  torch.backends.cudnn.deterministic = True


def generateCode(model, data_loader, num_data, bit, use_gpu=True):
  '''
  generate hash code for dataset
  '''  
  B = np.zeros([num_data, bit], dtype=np.float32)
  for image, _, ind in tqdm(data_loader,ascii=True):
    image = image.cuda()
    output = model(image)
    B[ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
  return B

def get_labels_str(label_txt_path):
  '''
    get labels (str) from txt file
  '''
  labels_int = np.loadtxt(label_txt_path, dtype=np.int64)
  labels_str = [''.join(label) for label in labels_int.astype(str)]
  labels_str = np.array(labels_str, dtype=str)
  return labels_str

def get_labels_int(label_txt_path):
  '''
    get labels (int) from txt file
  '''
  labels_int = np.loadtxt(label_txt_path, dtype=np.int64)
  return labels_int

def generateHash(model, samples):
  '''
    genrate hash code for a batch
  '''
  output = model(samples)
  B = torch.sign(output.cpu().data).numpy()
  return B

def get_scheduler(optimizer, opt):
  """
    Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
  """
  if opt.lr_policy == 'linear':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + 1 - opt.epoch) / float(opt.n_epochs_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer,step_size=opt.lr_decay_iters,gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,threshold=0.01,patience=5)
  elif opt.lr_policy == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.n_epochs,eta_min=0)
  else:
    return NotImplementedError(
        'learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler

def CalcSim(X, Y):
  S = (X.mm(Y.t()) > 0).float()
  return S

def log_trick(x):
  lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
      x, Variable(torch.FloatTensor([0.]).cuda()))
  return lt

def set_input_images(_input):
  _input = _input.cuda()
  _input = 2 * _input - 1
  return _input

def sample_img(image, sample_dir, name):
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
  image = image.cpu().detach()[0]
  image = transforms.ToPILImage()(image)
  image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)