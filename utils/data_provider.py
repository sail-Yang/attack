import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms
class HashingDataset_part(Dataset):
  '''
  图片路径和标签是分成两个txt的
  '''
  def __init__(self, data_path, txt_path, img_filename, label_filename, transform):
    self.img_path = data_path
    self.transform = transform
    # 获取图片路径列表
    img_filename_path = os.path.join(txt_path, img_filename)
    fp = open(img_filename_path, 'r')
    self.img_filename_list = [x.strip() for x in fp]
    fp.close()
    # 获取样本标签列表
    label_filepath = os.path.join(txt_path, label_filename)
    self.label_list = np.loadtxt(label_filepath, dtype=np.int64)

  def __getitem__(self, index):
    img = Image.open(os.path.join(self.img_path, self.img_filename_list[index]))
    img = img.convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    label = torch.from_numpy(self.label_list[index]).float()
    return img, label, index

  def __len__(self):
    return len(self.img_filename_list)

class HashingDataset(Dataset):
  '''
  图片路径和标签在一个txt里
  '''
  def __init__(self, data_path, txt_path, filename, transform):
    self.img_path = data_path
    self.transform = transform
    img_filename_path = os.path.join(txt_path, filename)
    fp = open(img_filename_path, 'r')
    self.img_list = [(self.img_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in fp]
    fp.close()
  
  def __getitem__(self, index):
    path, label = self.img_list[index]
    img = Image.open(path).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    return img, label, index
  
  def __len__(self):
    return len(self.img_filename_list)

def image_transform(resize_size, crop_size, is_train=True):
  '''
  transform设置
  '''
  return transforms.Compose([
      transforms.Resize(resize_size),
      transforms.CenterCrop(crop_size),
      transforms.ToTensor()
    ]
  )

def get_data(args):
  '''
  获取数据集的dataloader
  '''
  filepath = os.path.join(args.txt_path, "database.txt")
  if os.path.exists(filepath):
    dset_database = HashingDataset(args.data_path, args.txt_path, "database.txt",image_transform(args.resize_size, args.crop_size, False))
    dset_train = HashingDataset(args.data_path, args.txt_path, "train.txt",image_transform(args.resize_size, args.crop_size, True))
    dset_test = HashingDataset(args.data_path, args.txt_path, "test.txt",image_transform(args.resize_size, args.crop_size, False))
  else:
    dset_database = HashingDataset_part(args.data_path, args.txt_path, "database_img.txt","database_label.txt",image_transform(args.resize_size, args.crop_size, False))
    dset_train = HashingDataset_part(args.data_path, args.txt_path, "train_img.txt","train_label.txt",image_transform(args.resize_size, args.crop_size, True))
    dset_test = HashingDataset_part(args.data_path, args.txt_path, "test_img.txt","test_label.txt",image_transform(args.resize_size, args.crop_size, False))
  
  database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)
  return train_loader, test_loader, database_loader, num_train, num_test, num_database

def get_labels(args):
  pass