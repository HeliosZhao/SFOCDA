import numpy as np
import torch
from torch import nn
from torchvision import models
from ..utils import project_root
from .cpss import CPSS


class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


class DeeplabVGG(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DeeplabVGG, self).__init__()
        self.cfg = cfg
        vgg = models.vgg16(pretrained=False)
        # if pretrained:
        pretrain_path = 'pretrain/vgg16-00b39a1b-updated.pth'
        vgg.load_state_dict(torch.load(pretrain_path))
        print('load pretrained VGG')

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)
        self.CPSS_Module = CPSS(p=cfg.TRAIN.PROB, num_h=cfg.TRAIN.NUM_H, num_w=cfg.TRAIN.NUM_H)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_cpss(self, x):
        count = 0
        for submodule in self.features:
            x = submodule(x)
            if count == 3: 
                x = self.CPSS_Module(x)
            if count == 8: 
                x = self.CPSS_Module(x)

                x = x[0].unsqueeze(0)

            count += 1

        x = self.classifier(x)

        ## output x: 1,19,H,W
        return x


    def base_params(self, lr):
        return [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.classifier.parameters(), 'lr': lr * 10}]
                
    def optim_parameters(self, lr):
        if self.cfg.TRAIN.FREEZE_CLASSIFIER:
            for param in self.classifier.parameters():
                param.requires_grad = False
            print('-------------> Freeze Classifier <-------------------')
            return self.features.parameters()
        else:
            return self.parameters()


