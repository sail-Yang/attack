import torch
import os
import random
import numpy as np
from tqdm import tqdm

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