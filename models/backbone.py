import torch
import torch.nn as nn

from torchvision import models

class AlexNet(nn.Module):
  def __init__(self, num_bits):
    super().__init__()

    model = models.alexnet(weights="DEFAULT")
    self.features = model.features
    self.avgpool = model.avgpool
    hashing = []
    for i in range(6):
        hashing += [model.classifier[i]]
    hashing += [nn.Linear(4096, num_bits)]
    self.hashing = nn.Sequential(*hashing)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.hashing(x)
    return x


resnet_dict = {
  "ResNet18": models.resnet18,
  "ResNet34": models.resnet34,
  "ResNet50": models.resnet50,
  "ResNet101": models.resnet101,
  "ResNet152": models.resnet152,
}

class ResNet(nn.Module):
  def __init__(self, num_bits, res_model="ResNet50"):
    super(ResNet, self).__init__()
    model = resnet_dict[res_model](weights="DEFAULT")
    self.conv1 = model.conv1
    self.bn1 = model.bn1
    self.relu = model.relu
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4
    self.avgpool = model.avgpool
    self.feature_layers = nn.Sequential(
        self.conv1,
        self.bn1,
        self.relu,
        self.maxpool,
        self.layer1,
        self.layer2,
        self.layer3,
        self.layer4,
        self.avgpool,
    )
    self.hashing = nn.Linear(model.fc.in_features, num_bits)
    self.hashing.weight.data.normal_(0, 0.01)
    self.hashing.bias.data.fill_(0.0)

  def forward(self, x):
    x = self.feature_layers(x)
    x = torch.flatten(x, 1)
    x = self.hashing(x)
    return x

vgg_dict = {
  "vgg11":models.vgg11,
  "vgg13":models.vgg13,
  "vgg16":models.vgg16,
  "vgg19":models.vgg19,
  "vgg11bn":models.vgg11_bn,
  "vgg13bn":models.vgg13_bn,
  "vgg16bn":models.vgg16_bn,
  "vgg19bn":models.vgg19_bn
}

class VGG(nn.Module):
    def __init__(self, name, hash_bit):
        super(VGG, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features

        self.hash_layer = nn.Sequential()
        for i in range(6):
            self.hash_layer.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.hash_layer.add_module("hash", nn.Linear(model_vgg.classifier[6].in_features, hash_bit))

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.hash_layer(x)
        y = self.activation(x)
        return y

    def forward_factor(self, x, factor):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.hash_layer(x)
        y = self.activation(x * factor)
        return y