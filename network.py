import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets.folder import *

from gumbelmodule import *

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

        self.num_ftrs = 2048 * 1 * 1

        self.gate = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, 4),
        )

        self.max = nn.MaxPool2d((14,14))
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1):

        x = self.max(x)
        x = x.view(x.size(0), -1)

        weights_gate = self.gate(x)

        weights_gate = self.gs(weights_gate,temp=temperature, force_hard=True)

        return weights_gate
    
class AvgPools(nn.Module):
    def __init__(self):
        super(AvgPools, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d((4,4))  
        self.avg_pool2 = nn.AdaptiveAvgPool2d((5,5))
        self.avg_pool3 = nn.AdaptiveAvgPool2d((7,7))
        self.avg_pool4 = nn.AdaptiveAvgPool2d((11,11))

    def forward(self, x):

        f1 = self.avg_pool1(x)
        f2 = self.avg_pool2(x)
        f3 = self.avg_pool3(x)
        f4 = self.avg_pool4(x)

        return f1, f2, f3, f4


class Model_Ours(nn.Module):
    def __init__(self, model, feature_size=512, classes_num=200):
        super(Model_Ours, self).__init__()

        self.num_ftrs = 2048*1*1

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.Gate = Gate()
        self.avg_pool = AvgPools()

        self.expert_low = BasicConv(self.num_ftrs, self.num_ftrs, kernel_size=1, stride=1)

        self.expert_high = BasicConv(self.num_ftrs, self.num_ftrs, kernel_size=1, stride=1)

        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier_low = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        self.classifier_high = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs * 2),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs * 2, feature_size * 2),
            nn.BatchNorm1d(feature_size * 2),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size * 2, classes_num),
        )

    def forward(self, x):
         
        x = self.features(x)

        gate_weights = self.Gate(x)

        #low resolution feature
        batch_list = []
        for i in range(x.size(0)):
            f = list(self.avg_pool(x[i]))
            sample_list = []
            for j in range(4):
                f[j] = self.max(f[j]).unsqueeze(0)
                sample_list.append(f[j])
            sample_result = torch.cat(sample_list, dim=0)
            sample_pool = torch.sum(sample_result * gate_weights[i].view(-1, 1, 1, 1), dim=0).unsqueeze(0)
            batch_list.append(sample_pool)
        batch_pool = torch.cat(batch_list, dim=0)

        x_low = batch_pool
        x_low = self.expert_low(x_low)

        #high resolution feature
        x_high = self.expert_high(x)
        x_high = self.max(x)

        x_low = x_low.view(x_low.size(0), -1)
        x_high = x_high.view(x_high.size(0), -1)

        x_fusion = torch.cat((x_low,x_high), -1)

        output_low  = self.classifier_low(x_low)
        output_high = self.classifier_high(x_high)
        output = self.classifier(x_fusion)

        return output_low, output_high, output
