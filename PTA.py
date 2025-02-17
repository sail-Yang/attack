import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from models.hash_model import HashModel
from utils.config import load_config
from utils.data_provider import get_data,HashingDataset_part, image_transform
from utils.log import create_attack_hashing_logger
from utils.util import *
from utils.validate import CalcTopMap,CalcMap
import collections
import pandas as pd
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from models.AP_loss import SmoothAP
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def target_adv(query, model, pos_hash, neg_hash, pos_num=50, epsilon=8/255, iteration=20):
  delta = torch.zeros_like(query).cuda()
  noise = torch.rand_like(query).cuda()
  noise.data = noise.data.clamp(-epsilon, epsilon)
  noise.data = (query.data + noise.data).clamp(0, 1) - noise.data
  
  database_hash = torch.cat((pos_hash,neg_hash),0)
  s = epsilon/iteration
  delta.requires_grad = True
  lossObj = SmoothAP(0.1).requires_grad_()
  for i in (range(iteration)):
    noisy_query_hash = model((query+delta))
    loss = lossObj(noisy_query_hash,database_hash,pos_num)
    loss.backward()
    delta.data = delta - s * delta.grad.detach().sign()
    delta.data = delta.data.clamp(-epsilon, epsilon)
    delta.data = (query.data + delta.data).clamp(0, 1) - query.data
  delta.grad.zero_()
  return query + delta.detach(),query + noise.detach()
  
def evaluate_multi(args, model, database_hash):
  dset_test = HashingDataset_part(args.data_path, args.txt_path, "test_img.txt","test_label.txt",image_transform(args.resize_size, args.crop_size, False))
  sampler = SequentialSampler(dset_test)
  test_loader = DataLoader(dset_test, batch_size=1, num_workers=4, shuffle=False, sampler=sampler)
  
  # load database labels
  database_txt_path = os.path.join(args.txt_path, "database_label.txt")
  database_label_ = get_labels_int(database_txt_path)
  database_label = torch.from_numpy(database_label_).float()
  
  target_label_path = os.path.join(args.save_path, args.attack_method, "target_label_multi_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  if os.path.exists(target_label_path):
    target_labels = get_labels_int(target_label_path)
  candidate_labels = database_label.unique(dim=0)
  
  tmap_mean = 0
  tmap_mean_clean = 0
  perceptibility = 0
  for it, data in (enumerate(test_loader)):
    query = data[0].cuda()
    label = data[1]
    
    if it == 0:
      clean_labelL = label
    else:
      clean_labelL = torch.cat((clean_labelL, label))
    
    # target label
    if not os.path.exists(target_label_path):
      candi = []
      for iii in range(candidate_labels.shape[0]):
        if np.dot(candidate_labels[iii], label.numpy().reshape(args.n_class)) == 0 and torch.sum(candidate_labels[iii]) != 0:
          candi.append(iii)
      target_label_index = np.random.choice(range(len(candi)), size=1)
      target_label = candidate_labels.index_select(0, torch.from_numpy(np.array(candi)[target_label_index])).squeeze(0)
      target_index = np.where(target_label == 1)[0]
      
      if it == 0:
        L = torch.Tensor(target_label).cuda().reshape(1, len(target_label))
      else:
        L = torch.cat((L, torch.Tensor(target_label).cuda().reshape(1, len(target_label))))
    else:
      target_label = target_labels[it]
      target_index = np.where(target_label == 1)[0]
    
    res_true_label = label.reshape(args.n_class) - target_label
    res_true_label[target_index] = -(args.n_class - 1)
    coe = len(np.where(res_true_label == 1)[0])
    target_label_ = (-args.n_class - 1) * (torch.ones([args.n_class]) - target_label) + 1
    
    if it % 50 == 0:
      queryL = torch.Tensor(target_label).cuda().reshape(1, len(target_label))
    else:
      queryL = torch.cat((queryL, torch.Tensor(target_label).cuda().reshape(1, len(target_label))))
    
    pos_index = np.where(np.dot(target_label_, database_label_.transpose()) > 0)[0]
    neg_index = np.where((np.dot(res_true_label, database_label_.transpose()) > 0))[0]
    
    pos_index_ = np.random.choice(pos_index, args.pos_size, replace=args.replace)    
    try:
      neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 6), replace=args.replace)
    except:
      neg_index = np.where(np.dot(target_label, database_label_.transpose()) == 0)[0]
      neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 6), replace=args.replace)
    pos_train_hash = torch.Tensor(database_hash[pos_index_]).cuda()
    neg_train_hash = torch.Tensor(database_hash[neg_index_]).cuda() 
  
    a, _ = target_adv(query=query, model=model, pos_hash=pos_train_hash, neg_hash=neg_train_hash,
                      pos_num=pos_train_hash.shape[0], epsilon=args.epsilon, iteration=args.iteration)
    if it % 50 == 0:
      qB = torch.sign(model(a))
      qB_clean = torch.sign(model(data[0].cuda()))
    else:
      qB = torch.cat((qB, torch.sign(model(a))))
      qB_clean = torch.cat((qB_clean, torch.sign(model(data[0].cuda()))))
    
    if it % 50 == 49:
      queryL = queryL.cpu().detach().numpy()
      qB_ = qB.cpu().detach().numpy()
      qB_clean_ = qB_clean.cpu().detach().numpy()
      tmap = CalcTopMap(database_hash, qB_, database_label_, queryL, args.topK)
      tmap_clean = CalcTopMap(database_hash, qB_clean_, database_label_, queryL, args.topK)
      tmap_mean += tmap
      tmap_mean_clean += tmap_clean
    perceptibility += F.mse_loss(query, a).data * 1
  if not os.path.exists(target_label_path):
    np.savetxt(target_label_path, L.cpu().detach().numpy(), fmt='%d')
  
  qB_ = qB.cpu().detach().numpy()
  qB_clean_ = qB_clean.cpu().detach().numpy()
  tmap_clean_ori = CalcMap(database_hash, qB_, database_label_, clean_labelL)
  tmap_clean_ori_multi = CalcMap(database_hash, qB_clean_, database_label_, clean_labelL)
  
  tmap_mean = tmap_mean / (len(dset_test) / 50)
  tmap_mean_clean = tmap_mean_clean / (len(dset_test) / 50)
  
  logger.info("perceptibility: {:.7f}".format(torch.sqrt(perceptibility/len(dset_test))))
  logger.info("t-MAP of adv[retrieval database]: {:.7f}".format(tmap_mean))
  logger.info("t-MAP of origin[retrieval database]: {:.7f}".format(tmap_mean_clean))
  logger.info("MAP of adv[retrieval database]: {:.7f}".format(tmap_clean_ori))
  logger.info("MAP of origin[retrieval database]: {:.7f}".format(tmap_clean_ori_multi))
  
def evaluate(args, model, database_hash):
  dset_test = HashingDataset_part(args.data_path, args.txt_path, "test_img.txt","test_label.txt",image_transform(args.resize_size, args.crop_size, False))
  sampler = SequentialSampler(dset_test)
  test_loader = DataLoader(dset_test, batch_size=1, num_workers=4, shuffle=False, sampler=sampler)
  
  # load database lab
  database_txt_path = os.path.join(args.txt_path, "database_label.txt")
  database_label_ = get_labels_int(database_txt_path)
  database_label = torch.from_numpy(database_label_).float()
  if args.in_class:
    target = 1
    target_label_path = os.path.join(args.save_path, args.attack_method, "target_label_single_in_class_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  else:
    target = 0
    target_label_path = os.path.join(args.save_path, args.attack_method, "target_label_single_out_class_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  
  if os.path.exists(target_label_path):
    target_labels = get_labels_int(target_label_path)
  qB, qB_clean, clean_labelL, queryL = [], [], [], []  
  
  
  perceptibility = 0
  for it, data in (enumerate(tqdm(test_loader))):
    query = data[0].cuda()
    label = data[1]
    
    if not os.path.exists(target_label_path):
      target_index = get_single_target_label(label, target)
      target_label = np.zeros([args.n_class])
      target_label[target_index] = 1
    else:
      target_label = target_labels[it]
      target_index = np.where(target_label == 1)
    
    res_true_label = label.reshape(args.n_class) - target_label
    res_true_label[target_index] = -(args.n_class - 1)
    coe = len(np.where(res_true_label == 1)[0])
    
    pos_index = np.where(np.dot(target_label, database_label_.transpose()) == 1)[0]
    neg_index = np.where((np.dot(res_true_label, database_label_.transpose()) > 0))[0]
    
    pos_index_ = np.random.choice(pos_index, args.pos_size, replace=False)
    try:
      neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 4), replace=False)
    except:
      neg_index = np.where(np.dot(target_label, database_label_.transpose()) == 0)[0]
      neg_index_ = np.random.choice(neg_index, int(args.pos_size * coe * 4), replace=False)
    pos_train_hash = torch.Tensor(database_hash[pos_index_]).cuda()
    neg_train_hash = torch.Tensor(database_hash[neg_index_]).cuda()  
    
    a, _ = target_adv(query=query, model=model, pos_hash=pos_train_hash, neg_hash=neg_train_hash,
                      pos_num=pos_train_hash.shape[0], epsilon=args.epsilon, iteration=args.iteration)
   
    clean_labelL.append(label.cpu().detach())
    qB.append(torch.sign(model(a)).cpu().detach())
    qB_clean.append(torch.sign(model(query)).cpu().detach())
    queryL.append((torch.Tensor(target_label).reshape(1, len(target_label))).cpu().detach())
    
    perceptibility += F.mse_loss(query, a).data * 1
  
  qB = torch.cat(qB, 0).numpy()
  qB_clean = torch.cat(qB_clean, 0).numpy()
  queryL = torch.cat(queryL, 0).numpy()
  clean_labelL = torch.cat(clean_labelL, 0).numpy()
  
  if not os.path.exists(target_label_path):
    np.savetxt(target_label_path, queryL, fmt='%d')
  
  tmap_adv = CalcTopMap(database_hash, qB, database_label_, queryL, args.topK)
  tmap_ori = CalcTopMap(database_hash, qB_clean, database_label_, queryL, args.topK)
  map_adv = CalcTopMap(database_hash, qB, database_label_, clean_labelL, args.topK)
  map_ori = CalcTopMap(database_hash, qB_clean, database_label_, clean_labelL, args.topK)
  
  logger.info("perceptibility: {:.7f}".format(torch.sqrt(perceptibility/len(dset_test))))
  logger.info("t-MAP of adv[retrieval database]: {:.7f}".format(tmap_adv))
  logger.info("t-MAP of origin[retrieval database]: {:.7f}".format(tmap_ori))
  logger.info("MAP of adv[retrieval database]: {:.7f}".format(map_adv))
  logger.info("MAP of origin[retrieval database]: {:.7f}".format(map_ori))

def get_single_target_label(label, target):
  zero_index = np.where(label == target)
  zero_index = np.array(zero_index)
  random.seed(3)
  target_index = random.choice(zero_index[1])
  return target_index    
  
if __name__ == "__main__":
  conf_root = "./configs/pta.yaml"
  args = load_config(conf_root)
  seed_setting(args.seed)
  logger = create_attack_hashing_logger(args)
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  
  # load hash model
  hashModel = HashModel(args)
  hashModel.load_model()
  model = hashModel.model.cuda()
  
  # load database code
  database_code_path = os.path.join(args.save_path, args.attack_method, "database_code_{}_{}_{}_{}.txt".format(args.dataset, args.hash_model, args.backbone, args.num_bits))
  if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=np.float32)
  else:
    database_hash = generateCode(model, database_loader, num_database, args.num_bits)
    np.savetxt(database_code_path, database_hash, fmt="%d")
  
  if args.multi:
    logger.info("general target label...")
    evaluate_multi(args, model, database_hash)
  else:
    if args.in_class:
      logger.info("single in-class target label...")
    else:
      logger.info("single out-class target label...")
    evaluate(args, model, database_hash)
  