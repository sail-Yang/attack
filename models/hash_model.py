import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import hadamard
from tqdm import tqdm

from utils.validate import *
from .backbone import AlexNet,ResNet,VGG
import logging

class HashNetLoss(torch.nn.Module):
  '''
  Deep Hash method: HashNet
  '''
  def __init__(self, scale=1.0, alpha=1.0):
    super().__init__()
    self.scale = scale
    self.alpha = alpha
  
  def forward(self, u, y, ind):  # , scale):
    u = torch.tanh(u)
    s = (y @ y.t() > 0).float()
    sigmoid_alpha = self.alpha
    dot_product = sigmoid_alpha * u @ u.t()
    mask_positive = s > 0
    mask_negative = s < 1

    neg_log_probe = (
        torch.max(dot_product, torch.FloatTensor([0.0]).cuda())
        + torch.log(1 + torch.exp(-torch.abs(dot_product)))
    ) - s * dot_product

    s1 = torch.sum(mask_positive.float())
    s0 = torch.sum(mask_negative.float())
    s = s0 + s1

    neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * s / s1
    neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * s / s0

    loss = torch.sum(neg_log_probe) / s
    return loss


class CSQLoss(nn.Module):
  '''
  Deep Hash method: CSQ
  '''
  def __init__(self, num_classes, num_bits, lambda_=0.0001):
    super().__init__()
    self.lambda_ = lambda_
    hash_centers = self.get_hash_centers(num_classes, num_bits)
    self.hash_centers = hash_centers.cuda()
    random_center = torch.randint(0, 2, size=(1, num_bits)).float() * 2 - 1
    self.random_center = random_center.cuda()

  @staticmethod
  def get_hash_centers(num_classes, num_bits):
    if (num_bits & (num_bits - 1)) == 0 and num_classes <= 2 * num_bits:
      h_k = hadamard(num_bits)
      h_2k = np.concatenate((h_k, -h_k), 0)
      return torch.from_numpy(h_2k[:num_classes]).float()
    else:
      print("Generating Hash Centers...")
      for _ in range(1000):
        t = torch.randint(0, 2, (num_classes, num_bits)) * 2 - 1
        d = 2 * (num_bits - t @ t.T) - 1
        i = torch.triu_indices(num_classes, num_classes, 1)
        if torch.min(d[i]) >= num_bits / 3 and torch.mean(d[i]) >= num_bits / 2:
          break
    return t.float()

  def label2center(self, label):
    target_centers = (label @ self.hash_centers).sign()
    random_centers = self.random_center.expand(label.shape[0], -1)
    target_centers = torch.where(
      target_centers == 0, random_centers, target_centers
    )
    return target_centers

  def forward(self, u, y, ind):
    u = torch.tanh(u)
    c = self.label2center(y)
    c_loss = F.binary_cross_entropy((u + 1) / 2, (c + 1) / 2)
    q_loss = (u.abs() - 1.0).pow(2).mean()
    loss = c_loss + self.lambda_ * q_loss
    return loss

class DPSHLoss(nn.Module):
  '''
    Deep Hash method: DPSH
  '''
  def __init__(self, num_classes, num_train, num_bits, alpha):
    super().__init__()
    self.U = torch.zeros(num_train, num_bits).float().cuda()
    self.Y = torch.zeros(num_train, num_classes).float().cuda()
    self.alpha = alpha
  
  def forward(self, u, y, ind):
    self.U[ind, :] = u.data
    self.Y[ind, :] = y.float()
    
    s = (y @ self.Y.t() > 0).float()
    inner_product = u @ self.U.t() * 0.5
    
    likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
    likelihood_loss = likelihood_loss.mean()
    quantization_loss = self.alpha * (u - u.sign()).pow(2).mean()
    return likelihood_loss + quantization_loss

class HashModel:
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.model_name = args.hash_model
    self.build_model()
  
  def build_model(self):
    if "ResNet" in self.args.backbone:
      self.model = ResNet(self.args.num_bits,self.args.backbone)
    elif "VGG" in self.args.backbone:
      self.model = VGG(self.args.num_bits, self.args.backbone)
    elif "AlexNet" in self.args.backbone:
      self.model = AlexNet(self.args.num_bits)
    else:
      raise NotImplementedError
  
  def train(self, train_loader, val_loader, database_loader):
    num_epochs = self.args.epoch
    save_path = self.args.hash_save_path
    if self.model_name == "HashNet":
      alpha = self.args.hashnet_params.alpha
      critertion = HashNetLoss(alpha = alpha)
    elif self.model_name == "CSQ":
      lambda_ = self.args.csq_params.lambda_
      critertion = CSQLoss(self.args.n_class, self.args.num_bits, lambda_)
    elif self.model_name == "DPSH":
      critertion = DPSHLoss(self.args.n_class, self.args.dpsh_params.num_train, self.args.num_bits, self.args.dpsh_params.alpha)
    else:
      raise NotImplementedError
    
    model = self.model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=self.args.lr,
      betas=(0.9, 0.999),
    )
    
    best_mAP = 0
    for epoch in range(num_epochs):
      train_loss = 0
      for image, label, ind in tqdm(train_loader,ascii=True):
        image = image.cuda()
        label = label.cuda()
        
        output = model(image)
        optimizer.zero_grad()
        
        loss = critertion(output, label.float(), ind)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
      
      train_loss = train_loss / len(train_loader)
      logging.info(f"epoch: {epoch + 1} loss: {train_loss:.4f}")
      
      if (epoch + 1) % self.args.checkpoint == 0:
        map_, _ = validate_hash(val_loader, database_loader, model, self.args.topK)
        if map_ > best_mAP:
          best_mAP = map_
          # save the paramters of the model
          file_name = f"{self.model_name}_{self.args.backbone}_{self.args.dataset}_{str(self.args.num_bits)}.pt"
          dir_path = os.path.join(save_path, self.model_name)
          if not os.path.exists(dir_path):
            os.makedirs(dir_path)
          path = os.path.join(save_path, self.model_name, file_name)
          torch.save(model.state_dict(), path)

        logging.info(f"mAP: {map_: .4f} best_mAP: {best_mAP:.4f}")
  
  def load_model(self):
    file_name = (
      f"{self.model_name}_{self.args.backbone}_{self.args.dataset}_{str(self.args.num_bits)}.pt"
    )
    save_path = self.args.hash_save_path
    path = os.path.join(save_path, self.args.hash_model, file_name)
    checkpoint = torch.load(path)
    self.model.load_state_dict(checkpoint)

  @staticmethod
  def load_t_model(model_path):
    '''
    load attacked model
    '''
    model = torch.load(model_path)
    if torch.cuda.is_available():
      model = model.cuda()
    model.eval()
    return model

  def test_model(self, test_loader, database_loader):
    model = self.model.eval().cuda()
    map_, data = validate_hash(test_loader, database_loader, model, top_k=5000)
    logging.info(f"Test mAP: {map_:.4f}")    