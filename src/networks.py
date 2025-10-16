import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNet_AT(nn.Module):
    def __init__(self, block, layers,featureVectoreSize):
        super(ResNet_AT, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, featureVectoreSize, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=x.permute(0,4,1,2,3)

        batch_size, seqlen, nc, h, w = x.size()

        x = x.reshape(-1, nc, h, w)
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)

        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        #print("here reshaping")
        out = f.reshape(batch_size, seqlen, -1)
        #out=f
        return out

def resnet18_at(featureVectoreSize, **kwargs):
    # Constructs base a ResNet-18 model.
    model = ResNet_AT(BasicBlock, [2, 2, 2, 2],featureVectoreSize, **kwargs)
    return model

def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model

class Network(nn.Module):
    def __init__(self,Networkstruct):
        super(Network, self).__init__()
        self.network=[]
        
        for net in Networkstruct:            
            if(net=='Resnet'):
                _structure=resnet18_at(Networkstruct[net]['featureVectoreSize'])
                _parameterDir = Networkstruct[net]['path']
                self.backbone = model_parameters(_structure, _parameterDir)
            if(net=='GRU'):
                inputGru=Networkstruct[net]['input_size']
                num_layers=Networkstruct[net]['num_layers']
                hidden_size=Networkstruct[net]['hidden_size']
                self.gru=nn.GRU(inputGru,hidden_size,num_layers,batch_first=True,bidirectional=Networkstruct[net]['bidirectional'])

            if(net=='FC1'):
                
                self.pred_fc1 = nn.Linear(Networkstruct[net]['input_size'], Networkstruct[net]['output_size'])
            if(net=='ReLU1'):
                self.relu1 = torch.nn.ReLU()
            if(net=='ReLU2'):
                self.relu2 = torch.nn.ReLU()
            if(net=='ReLU3'):
                self.relu3 = torch.nn.ReLU()   
            if(net=='FC2'):
                self.pred_fc2 = nn.Linear(Networkstruct[net]['input_size'], Networkstruct[net]['output_size'])
            if(net=='FC3'):
                self.pred_fc3 = nn.Linear(Networkstruct[net]['input_size'], Networkstruct[net]['output_size'])
            if(net=='FC4'):
                self.pred_fc4 = nn.Linear(Networkstruct[net]['input_size'], Networkstruct[net]['output_size'])
            if(net=='Transformer'):
                self.tranformer=None
                
            if(net=='DropOut1'):
                self.dropout1=nn.Dropout(Networkstruct[net]['p1'])
                
            if(net=='DropOut2'):
                self.dropout2=nn.Dropout(Networkstruct[net]['p2'])
            if(net=='DropOut3'):
                self.dropout3=nn.Dropout(Networkstruct[net]['p3'])
            if(net=='Sigmoid'):
                self.sigmoid=nn.Sigmoid()
            self.network.append(net)

    def forward(self, x):
        for net in self.network:
            if(net=='Resnet'):
                x=self.backbone(x)
                continue
            if(net=='GRU'):
                #print(x.size())
                x,_=self.gru(x)
                continue
            if(net=="Mean"):
                x=x.mean(dim=1)
                continue
              
            if(net=='FC1'):
                x=torch.squeeze(x)
                #print(x.size())
                
                x=self.pred_fc1(x)
                continue
            if(net=='FC2'):
                x=self.pred_fc2(x)
                continue
            if(net=='FC3'):
                x=self.pred_fc3(x)
                continue
            if(net=='tranformer'):
                x=self.tranformer(x)
                continue
            if(net=='ReLU1'):
                x=self.relu1(x)
                continue
            if(net=='ReLU2'):
                x=self.relu2(x)
                continue
            if(net=='ReLU3'):
                x=self.relu3(x)
                continue
            if(net=='Sigmoid'):
                x=self.sigmoid(x)
                continue
            if(net=='DropOut1'):
                x=self.dropout1(x)
                continue
            if(net=='DropOut2'):
                x=self.dropout2(x)
                continue
            if(net=='DropOut3'):
                x=self.dropout3(x)
                continue
            if(net=='FC4'):
                x = self.pred_fc4(x)
                continue
            raise NameError("module name not matched"+net)
        return x
    

