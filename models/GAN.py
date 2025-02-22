import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .spectral_norm import spectral_norm as SpectralNorm

class Discriminator(nn.Module):
  """
    Discriminator network with PatchGAN.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
  """
  def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
    super(Discriminator, self).__init__()
    layers = []
    layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01))

    curr_dim = conv_dim
    for i in range(1, repeat_num):
      layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01))
      curr_dim = curr_dim * 2

    kernel_size = int(image_size / (2**repeat_num))
    self.main = nn.Sequential(*layers)
    self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

  def forward(self, x):
    h = self.main(x)
    out = self.fc(h)
    return out.squeeze()

class Generator(nn.Module):
  """
    Generator: Encoder-Decoder Architecture.
    Reference: https://github.com/yunjey/stargan/blob/master/model.py
  """
  def __init__(self):
    super(Generator, self).__init__()
    # Label Encoder
    self.label_encoder = LabelEncoder()
    
    # Image Encoder
    curr_dim = 64
    image_encoder = [
      nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
      nn.InstanceNorm2d(curr_dim),
      nn.ReLU(inplace=True)
    ]
    
    # Down Sampling
    for i in range(2):
      image_encoder += [
        nn.Conv2d(curr_dim, 
                  curr_dim * 2, 
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  bias=True),
        nn.InstanceNorm2d(curr_dim * 2),
        nn.ReLU(inplace=True)
      ]
      curr_dim = curr_dim * 2
    # Bottleneck
    for i in range(3):
      image_encoder += [
        ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
      ]
    self.image_encoder = nn.Sequential(*image_encoder)
    
    
    # Decoder
    decoder = []
    # Bottleneck
    for i in range(3):
      decoder += [
        ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t')
      ]
    
    # Up Sampling
    for i in range(2):
      decoder += [
        nn.ConvTranspose2d(curr_dim,
                           curr_dim // 2,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.InstanceNorm2d(curr_dim // 2),
        nn.ReLU(inplace=True)
      ]
      curr_dim = curr_dim // 2
    self.residual = nn.Sequential(
      nn.Conv2d(curr_dim + 3,
                  3,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  bias=False),
      nn.Tanh()
    )
    self.decoder = nn.Sequential(*decoder)
  
  def forward(self, x, label_feature):
    mixed_feature = self.label_encoder(x,label_feature)
    encode = self.image_encoder(mixed_feature)
    decode = self.decoder(encode)
    decode_x = torch.cat([decode, x], dim=1)
    adv_x = self.residual(decode_x)
    return adv_x, mixed_feature

class LabelEncoder(nn.Module):
  def __init__(self, nf=128):
    super(LabelEncoder, self).__init__()
    self.nf = nf
    curr_dim = nf
    self.size = 14
    
    self.fc = nn.Sequential(
      nn.Linear(512, curr_dim * self.size * self.size),
      nn.ReLU(True)
    )
    
    transform = []
    for i in range(4):
      transform += [
        nn.ConvTranspose2d(curr_dim,
                           curr_dim // 2,
                           kernel_size=4,
                           stride=2,
                           padding=1,
                           bias=False),
        nn.InstanceNorm2d(curr_dim // 2, affine=False),
        nn.ReLU(inplace=True)
      ]
      curr_dim = curr_dim // 2
    
    transform += [
      nn.Conv2d(curr_dim,
                  3,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                bias=False)
    ]
    self.transform = nn.Sequential(*transform)
  
  def forward(self, image, label_feature):
    label_feature = self.fc(label_feature)
    label_feature = label_feature.view(label_feature.size(0), self.nf, self.size, self.size)
    label_feature = self.transform(label_feature)

    mixed_feature = torch.cat((label_feature, image), dim=1)
    return mixed_feature


class ResidualBlock(nn.Module):
  """
    Residual Block.
  """
  def __init__(self, dim_in, dim_out, net_mode=None):
    if net_mode == 'p' or (net_mode is None):
      use_affine = True
    elif net_mode == 't':
      use_affine = False
    super(ResidualBlock, self).__init__()
    self.main = nn.Sequential(
                    nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), 
                    nn.InstanceNorm2d(dim_out,affine=use_affine),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), 
                    nn.InstanceNorm2d(dim_out,affine=use_affine)
                )
  def forward(self, x):
    return x + self.main(x)

class GANLoss(nn.Module):
  """
    Define different GAN objectives
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input
  """
  def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
    """ Initialize the GANLoss class
    Parameters:
      gan_mode (str) -- the type of GAN objective.(vanilla, lsgan, wgangp)
      target_real_label (bool) -- label of a real image
      target_fake_label (bool) -- label of a fake image
    Notes:
      Do not use sigmoid as the last layer of Discriminator.
      LSGAN needs no sigmoid.
      vanilla GANs will handle it with BCEWithLogitsLoss.
    """
    super(GANLoss, self).__init__()
    self.register_buffer('real_label',torch.tensor(target_real_label))
    self.register_buffer('fake_label',torch.tensor(target_fake_label))
    self.gan_mode = gan_mode
    if gan_mode == 'lsgan':
      self.loss = nn.MSELoss()
    elif gan_mode == 'vanilla':
      self.loss = nn.BCEWithLogitsLoss()
    elif gan_mode in ['wgangp']:
      self.loss = None
    else:
      raise NotImplementedError('gan mode %s not implemented' % gan_mode)
  
  def get_target_tensor(self, prediction, target_is_real):
    """
      Create label tensors with the same size as the input
      Parameters:
        prediction (tensor) -- the prediction from a discriminator
        target_is_real (bool) -- if the ground truth label is for real images or fake images
      Returns:
        A label tensor filled with ground truth label and the size of the input
    """
    if target_is_real:
      real_label = self.real_label.expand(prediction.size(0),1)
      target_tensor = torch.cat([prediction, real_label], dim=-1)
    else:
      fake_label = self.fake_label.expand(prediction.size(0),1)
      target_tensor = torch.cat([prediction, fake_label], dim=-1)
    return target_tensor
  
  def __call__(self, prediction, label, target_is_real):
    """
      Calculate loss given Discriminator's output and grount truth labels.
      Parameters:
        prediction (tensor) - - tpyically the prediction output from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images
      Returns:
        the calculated loss.
    """
    if self.gan_mode in ['lsgan', 'vanilla']:
      target_tensor = self.get_target_tensor(label, target_is_real)
      loss = self.loss(prediction, target_tensor)
    elif self.gan_mode == 'wgangp':
      if target_is_real:
        loss = -prediction.mean()
      else:
        loss = prediction.mean()
    return loss