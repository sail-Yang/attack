import torch.nn as nn
import torch

class PrototypeNet(nn.Module):
  def __init__(self, bit, num_classes):
    super(PrototypeNet, self).__init__()

    self.feature = nn.Sequential(
                      nn.Linear(num_classes, 4096),
                      nn.ReLU(True), 
                      nn.Linear(4096, 512)
                    )
    self.hashing = nn.Sequential(nn.Linear(512, bit), nn.Tanh())
    self.classifier = nn.Sequential(nn.Linear(512, num_classes),nn.Sigmoid())

  def forward(self, label):
    f = self.feature(label)
    h = self.hashing(f)
    c = self.classifier(f)
    return f, h, c

  def load_model(self, pnet_path):
    checkpoint = torch.load(pnet_path)
    self.load_state_dict(checkpoint)
    self.eval()
  