import torch
import time
import torch.nn as nn
from omegaconf import OmegaConf
from utils.config import load_config
from utils.data_provider import get_data
from utils.log import create_attack_hashing_logger
from utils.util import *
from utils.validate import CalcTopMap
from models.GAN import *
from models.prototypeNet import PrototypeNet
from models.hash_model import HashModel

torch.multiprocessing.set_sharing_strategy("file_system")

class TargetAttackGAN(nn.Module):
  def __init__(self, args):
    super(TargetAttackGAN, self).__init__()
    self.bit = args.num_bits
    self.n_class = args.n_class
    rec_weight_dic = {'FLICKR-25K': 100, 'NUS-WIDE':50, 'MS-COCO': 50}
    self.rec_w = rec_weight_dic[args.dataset]
    self.dis_w = 1
    self.batch_size = args.batch_size
    self.lr = args.lr
    self.args = args
    self.model_name = '{}_{}_{}_{}'.format(args.dataset, args.hash_model, args.backbone, args.num_bits)
    self.t_model_name = '{}_{}_{}_{}'.format(args.dataset, args.trans_config.t_hash_model, args.trans_config.t_backbone, args.trans_config.t_bit)
    self.build_model()
  
  def build_model(self):
    self.generator = Generator().cuda()
    self.discriminator = Discriminator(self.n_class).cuda()
    self.prototype_net = PrototypeNet(self.bit, self.n_class).cuda()
    # load hash model
    self.hashModel = HashModel(self.args, args.hash_model, args.backbone, args.num_bits)
    self.hashModel.load_model()
    self.hashModel = self.hashModel.model.cuda()
    
    self.criterionGAN = GANLoss(self.args.gan_mode).cuda()
  
  def set_requires_grad(self, nets, requires_grad=False):
    """
      Set requies_grad=Fasle for all the networks to avoid unnecessary computations
      Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad
  
  def train(self, train_loader, target_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test):
    # L2 loss function
    criterion_l2 = torch.nn.MSELoss()
    # optimizers
    optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
    self.optimizers = [optimizer_gen, optimizer_dis]
    self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
    
    # prototype net
    prototype_path = os.path.join(self.args.save_path, self.args.attack_method, 'prototypenet_{}.pt'.format(self.model_name))
    if os.path.exists(prototype_path):
      self.load_module(self.prototype_net, prototype_path)
    else:
      self.train_prototype_net(database_loader, target_labels, train_labels, num_train)
    self.prototype_net.eval()
    self.test_prototype(target_labels, database_loader, database_labels, num_database, num_test)
  
    total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
    for epoch in range(self.args.epoch_count, total_epochs):
      logger.info('Train epoch: {}, learning rate: {:.7f}'.format(epoch, self.lr))
      for i, data in enumerate(train_loader):
        real_input, batch_label, batch_ind = data
        real_input = self.set_input_images(real_input)
        batch_label = batch_label.cuda()
        
        select_index = np.random.choice(range(target_labels.size(0)), size=batch_label.size(0))
        batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        
        feature, target_hash_l, _ = self.prototype_net(batch_target_label)
        target_hash_l = torch.sign(target_hash_l.detach())
        fake_g, _ = self.generator(real_input, feature.detach())
        
        # Update D
        if i % 3 == 0:
          self.set_requires_grad(self.discriminator, True)
          optimizer_dis.zero_grad()
          real_d = self.discriminator(real_input)
          # stop backprop to the generator by detaching
          fake_d = self.discriminator(fake_g.detach())
          real_d_loss = self.criterionGAN(real_d, batch_label, True)
          fake_d_loss = self.criterionGAN(fake_d, batch_target_label, False)
          d_loss = (real_d_loss + fake_d_loss) / 2
          d_loss.backward()
          optimizer_dis.step()
        
        # Update G
        self.set_requires_grad(self.discriminator, False)
        optimizer_gen.zero_grad()
        fake_g_d = self.discriminator(fake_g)
        fake_g_loss = self.criterionGAN(fake_g_d, batch_target_label, True)
        reconstruction_loss = criterion_l2(fake_g, real_input)
        target_hashing_g = self.hashModel((fake_g + 1) / 2)
        logloss = target_hashing_g * target_hash_l
        logloss = torch.mean(logloss)
        logloss = (-logloss + 1)
        
        # backpropagation
        g_loss = self.rec_w * reconstruction_loss + 5*logloss + self.dis_w*fake_g_loss
        g_loss.backward()
        optimizer_gen.step()
        
        if i % self.args.sample_checkpoint == 0:
          dir_path = os.path.join(self.args.save_path, self.args.attack_method, "sample")
          sample_img(fake_g, dir_path, str(epoch) + '_' + str(i) + '_fake')
          sample_img(real_input, dir_path, str(epoch) + '_' + str(i) + '_real')
        
        if i % self.args.checkpoint == 0:
          
          logger.info(
            'step: {:3d} g_loss: {:.3f} d_loss: {:.3f} hash_loss: {:.3f} r_loss: {:.7f}'
                        .format(i, fake_g_loss, d_loss, logloss, reconstruction_loss)
          )
      self.update_learning_rate()
    self.save_generator()
  
  def test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
    self.prototype_net.eval()
    self.generator.eval()
    targeted_labels = np.zeros([num_test, self.n_class])
    qB = np.zeros([num_test, self.bit])
    
    perceptibility = 0
    start = time.time()
    for it, data in enumerate(test_loader):
      data_input, _, data_ind = data
      select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
      batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
      targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()
      
      data_input = self.set_input_images(data_input)
      feature,_,_ = self.prototype_net(batch_target_label.cuda())
      target_fake, _ = self.generator(data_input, feature)
      target_fake = (target_fake + 1) / 2
      data_input = (data_input + 1) / 2 
      
      target_hashing = self.hashModel(target_fake)
      qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()
      
      perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)
    end = time.time()
    logger.info('Running time: %s Seconds'%(end-start))
    np.savetxt(os.path.join(self.args.save_path, self.args.attack_method, "test_code_{}.txt".format(self.model_name)), qB, fmt="%d")
    np.savetxt(os.path.join(self.args.save_path, self.args.attack_method, "target_label_{}.txt".format(self.model_name)), targeted_labels, fmt="%d")
    # load database code
    database_code_path = os.path.join(self.args.save_path, self.args.attack_method, "database_code_{}.txt".format(self.model_name))
    if os.path.exists(database_code_path):
      database_hash = np.loadtxt(database_code_path, dtype=np.float32)
    else:
      database_hash = generateCode(self.hashModel, database_loader, num_database, args.num_bits)
      np.savetxt(database_code_path, database_hash, fmt="%d")
    database_txt_path = os.path.join(self.args.txt_path, "database_label.txt")
    database_labels_int = get_labels_int(database_txt_path)
    logger.info(f"perceptibility: {torch.sqrt(perceptibility/num_test):.7f}")
    t_map = CalcTopMap(database_hash, qB, database_labels_int, targeted_labels, topk=self.args.topK)
    logger.info('t_MAP(retrieval database): %3.5f' % (t_map))
    map_ = CalcTopMap(database_hash, qB, database_labels_int, test_labels, topk=self.args.topK)
    logger.info('MAP(retrieval database): %3.5f' % (map_))
  
  def transfer_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
    # load target hash model
    self.t_hashModel = HashModel(self.args, args.trans_config.t_hash_model, args.trans_config.t_backbone, args.trans_config.t_bit)
    self.t_hashModel.load_model()
    self.t_hashModel = self.t_hashModel.model.cuda()
    self.bit = self.args.trans_config.t_bit
    
    self.generator.eval()
    self.prototype_net.eval()
    
    qB = np.zeros([num_test, self.bit])
    targeted_labels = np.zeros([num_test, self.n_class])
    
    perceptibility = 0
    start = time.time()
    for it, data in enumerate(test_loader):
      data_input, _, data_ind = data
      select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
      batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
      targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()
      
      data_input = self.set_input_images(data_input)
      feature,_,_ = self.prototype_net(batch_target_label.cuda())
      target_fake, _ = self.generator(data_input, feature)
      target_fake = (target_fake + 1) / 2
      data_input = (data_input + 1) / 2 
      
      target_hashing = self.t_hashModel(target_fake)
      qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()
      
      perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)
    end = time.time()
    logger.info('Running time: %s Seconds'%(end-start))
    np.savetxt(os.path.join(self.args.save_path, self.args.attack_method, "test_code_{}.txt".format(self.t_model_name)), qB, fmt="%d")
    np.savetxt(os.path.join(self.args.save_path, self.args.attack_method, "target_label_{}.txt".format(self.t_model_name)), targeted_labels, fmt="%d")
    # load database code
    database_code_path = os.path.join(self.args.save_path, self.args.attack_method, "database_code_{}.txt".format(self.model_name))
    if os.path.exists(database_code_path):
      database_hash = np.loadtxt(database_code_path, dtype=np.float32)
    else:
      database_hash = generateCode(self.t_hashModel, database_loader, num_database, args.num_bits)
      np.savetxt(database_code_path, database_hash, fmt="%d")
    database_txt_path = os.path.join(self.args.txt_path, "database_label.txt")
    database_labels_int = get_labels_int(database_txt_path)
    logger.info(f"perceptibility: {torch.sqrt(perceptibility/num_test):.7f}")
    t_map = CalcTopMap(database_hash, qB, database_labels_int, targeted_labels, topk=self.args.topK)
    logger.info('t_MAP(retrieval database): %3.5f' % (t_map))
    map_ = CalcTopMap(database_hash, qB, database_labels_int, test_labels, topk=self.args.topK)
    logger.info('MAP(retrieval database): %3.5f' % (map_))
    
            
  def train_prototype_net(self,database_loader, target_labels, train_labels, num_train):
    optimizer_l = torch.optim.Adam(self.prototype_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
    epochs = 100
    steps = 300
    batch_size = 64
    lr_steps = epochs * steps
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    criterion_l2 = torch.nn.MSELoss()
    
    # hash codes of training set
    B = generateCode(self.hashModel, train_loader, num_train, self.bit)
    B = torch.from_numpy(B).cuda().requires_grad_(True)
    
    for epoch in range(epochs):
      for i in range(steps):
        select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
        batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()
        optimizer_l.zero_grad()
        
        _, target_hash_l, label_pred = self.prototype_net(batch_target_label)
        theta_x = target_hash_l.mm(B.t()) / 2        
        S = CalcSim(batch_target_label.cpu(), train_labels)
        logloss = (S.cuda() * theta_x - log_trick(theta_x)).sum() / (num_train * batch_size)
        logloss = -logloss
        regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
        classifer_loss = criterion_l2(label_pred, batch_target_label)
        loss = logloss + classifer_loss + regterm
        loss.backward()
        optimizer_l.step()
        if i % self.args.checkpoint == 0:
          logger.info('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}, l2_loss: {:.7f}'.format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm, classifer_loss))
        scheduler.step()
  
    self.save_prototypenet()
    
  def load_module(self, model, model_path):
    model.load_state_dict(torch.load(model_path))
    
  def save_prototypenet(self):
    prototype_path = os.path.join(self.args.save_path, self.args.attack_method, 'prototypenet_{}.pt'.format(self.model_name))
    torch.save(self.prototype_net.state_dict(), prototype_path)
  
  def save_generator(self):
    gen_path = os.path.join(self.args.save_path, self.args.attack_method, 'generator_{}.pt'.format(self.model_name))
    torch.save(self.generator.state_dict(), gen_path)
  
  def load_all_model(self):
    prototype_path = os.path.join(self.args.save_path, self.args.attack_method, 'prototypenet_{}.pt'.format(self.model_name))
    gen_path = os.path.join(self.args.save_path, self.args.attack_method, 'generator_{}.pt'.format(self.model_name))
    self.load_module(self.prototype_net, prototype_path)
    self.load_module(self.generator, gen_path)
    
  def test_prototype(self, target_labels, database_loader, database_labels, num_database, num_test):
    targeted_labels = np.zeros([num_test, self.n_class])
    qB = np.zeros([num_test, self.bit])
    
    for i in range(num_test):
      select_index = np.random.choice(range(target_labels.size(0)), size=1)
      batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
      targeted_labels[i, :] = batch_target_label.numpy()[0]
      
      _, target_hash_l, __ = self.prototype_net(batch_target_label.cuda().float())
      qB[i, :] = torch.sign(target_hash_l.cpu().data).numpy()[0]
    
    # load database code
    database_code_path = os.path.join(self.args.save_path, self.args.attack_method, "database_code_{}.txt".format(self.model_name))
    if os.path.exists(database_code_path):
      database_hash = np.loadtxt(database_code_path, dtype=np.float32)
    else:
      database_hash = generateCode(self.hashModel, database_loader, num_database, args.num_bits)
      np.savetxt(database_code_path, database_hash, fmt="%d")
    database_txt_path = os.path.join(self.args.txt_path, "database_label.txt")
    database_labels_int = get_labels_int(database_txt_path)
    t_map = CalcTopMap(database_hash, qB, database_labels_int, targeted_labels, topk=self.args.topK)
    logger.info('t_MAP(retrieval database): %3.5f' % (t_map))
    
  def set_input_images(self, _input):
    _input = _input.cuda()
    _input = 2 * _input - 1
    return _input
        
  def update_learning_rate(self):
    """Update learning rates for all the networks; called at the end of every epoch"""
    for scheduler in self.schedulers:
      if self.args.lr_policy == 'plateau':
        scheduler.step(0)
      else:
        scheduler.step()
    self.lr = self.optimizers[0].param_groups[0]['lr']
    
     
if __name__ == "__main__":
  conf_root = "./configs/pros_gan.yaml"
  args = load_config(conf_root)
  seed_setting(args.seed)
  logger = create_attack_hashing_logger(args)
  
  train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
  train_txt_path = os.path.join(args.txt_path, "train_label.txt")
  database_txt_path = os.path.join(args.txt_path, "database_label.txt")
  test_txt_path = os.path.join(args.txt_path, "test_label.txt")
  train_labels = torch.from_numpy(get_labels_int(train_txt_path)).float()
  database_labels = torch.from_numpy(get_labels_int(database_txt_path)).float()
  test_labels = torch.from_numpy(get_labels_int(test_txt_path)).float()
  target_labels = database_labels.unique(dim=0)
  
  model = TargetAttackGAN(args=args)
  if args.train:
    logger.info("training...")
    model.train(train_loader, target_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test)
  
  if args.test:
    logger.info("testing...")
    model.load_all_model()
    model.test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
  
  if args.transfer:
    logger.info("transfer testing ...")
    logger.info("target model: {} {} {}".format(args.trans_config.t_hash_model, args.trans_config.t_backbone, args.trans_config.t_bit))
    model.load_all_model()
    model.transfer_test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)