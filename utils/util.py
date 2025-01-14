import torch
import os
import random
import numpy as np

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