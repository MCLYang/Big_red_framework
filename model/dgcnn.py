import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from time import time
import torch.nn.functional as F
import pdb

def get_model(input_channel,is_synchoization = 'Instance'):
    return(DGCNN_semseg(num_channel=input_channel,synchoization = is_synchoization))

def get_model_name(input_channel):
    return("dgcnn_"+str(input_channel)+"c")

def get_loss(input_channel):
    return(dgcnn_loss())


class dgcnn_loss(nn.Module):
    def __init__(self):
        super(dgcnn_loss, self).__init__()
        self.weight = torch.tensor([1,1]).cuda()
    def load_weight(self,weight):
        self.weight = torch.tensor(weight).cuda()
    # def forward(self, pred, target, trans_feat, weight):
    def forward(self, pred, target):
        pred = pred[0]
        weight = self.weight
        target = target.view(-1)
        pred = pred.reshape(-1,2)
        loss = F.cross_entropy(pred, target,weight = weight, reduction='mean')
        return loss

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

#idx = knn(x[:, :3], k=k)   # (batch_size, num_points, k)

def get_graph_feature(x, k=20, idx=None, moreFeatures=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x[:, :3], k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature




class DGCNN_semseg(nn.Module):
    def __init__(self,num_channel = 4,synchoization='Instance'):
        super(DGCNN_semseg, self).__init__()

        self.emb_dims = 1024
        self.k = 20
        self.drop = 0.5
        if(synchoization=='Instance'):
            self.bn1 = nn.InstanceNorm2d(64)
            self.bn2 = nn.InstanceNorm2d(64)
            self.bn3 = nn.InstanceNorm2d(64)
            self.bn4 = nn.InstanceNorm2d(64)
            self.bn5 = nn.InstanceNorm2d(64)
            self.bn6 = nn.InstanceNorm1d(self.emb_dims)
            self.bn7 = nn.InstanceNorm1d(512)
            self.bn8 = nn.InstanceNorm1d(256)
        elif(synchoization=='BN'):
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(64)
            self.bn5 = nn.BatchNorm2d(64)
            self.bn6 = nn.BatchNorm1d(self.emb_dims)
            self.bn7 = nn.BatchNorm1d(512)
            self.bn8 = nn.BatchNorm1d(256)


        self.conv1 = nn.Sequential(nn.Conv2d(int(num_channel*2), 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.drop)
        self.conv9 = nn.Conv1d(256, 2, kernel_size=1, bias=False)
        

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, moreFeatures=True)   # (batch_size, num_channel, num_points) -> (batch_size, num_channel*2, num_points, k)
        x = self.conv1(x)                       # (batch_size,num_channel*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = x.transpose(2, 1)

        return [x]