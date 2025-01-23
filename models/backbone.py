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
        # model = resnet_dict[res_model](weights="DEFAULT")
        model = resnet_dict[res_model](pretrained=True)
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

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
class VGG(nn.Module):
    def __init__(self, bit, model_name="VGG11"):
        super(VGG, self).__init__()
        original_model = vgg_dict[model_name](pretrained=True)
        self.features = original_model.features
        self.cl1 = nn.Linear(25088, 4096)
        self.cl1.weight = original_model.classifier[0].weight
        self.cl1.bias = original_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[3].weight
        cl2.bias = original_model.classifier[3].bias

        self.classifier = nn.Sequential(
            self.cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        # x = (x-self.mean)/self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y