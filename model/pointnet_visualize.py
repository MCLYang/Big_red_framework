from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb



def get_model(input_channel,is_synchoization = 'BN'):
    return(Pointnet_sem_seg(num_channel=input_channel))

def get_model_name(input_channel):
    return("pointnet_"+str(input_channel)+"c")

def get_loss(input_channel):
    return(pointnet_loss(num_channel = input_channel))


class pointnet_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001,feature_transform = True,num_channel = 4):
        super(pointnet_loss, self).__init__()
        if(num_channel>3):
            feature_transform = True
        else:
            feature_transform = False
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.feature_transform = feature_transform
        self.weight = torch.tensor([1,1])

    def load_weight(self,weight):
        self.weight = torch.tensor(weight).cpu()
    def forward(self, pred_mics, target):
        weight = self.weight
        target = target.view(-1).cpu()
        pred = pred_mics[0].cpu()
        trans_feat = pred_mics[1].cpu()
        pred = pred.view(-1,2)
        # pdb.set_trace()
        if(self.feature_transform == True):
            loss = F.nll_loss(pred, target, weight = weight)
            mat_diff_loss = feature_transform_regularizer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        else:
            total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


def feature_transform_regularizer(trans):
    #pdb.set_trace()
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    product =  torch.bmm(trans, trans.transpose(2,1)) - I
    product = product.cpu()
    loss = torch.mean(torch.norm(product, dim=(1,2)))
    return loss


class STN3d(nn.Module):
    def __init__(self,num_channal = 5):
        super(STN3d, self).__init__()
        self.num_channal = num_channal
        self.conv1 = torch.nn.Conv1d(num_channal, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)

        #self.bn4 = nn.BatchNorm1d(512)

        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):

        batchsize = x.size()[0]
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # print(x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=False, feature_transform=True, channel=5):
        super(PointNetEncoder, self).__init__()
        self.channel = channel
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        #pdb.set_trace()

        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        
        #pdb.set_trace()
        #print(trans_feat.shape)
        pointfeat = x
        #pdb.set_trace()
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        pdb.set_trace()
        x = torch.max(x, 2, keepdim=True)[0]
        pdb.set_trace()
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            #pdb.set_trace()
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            #pdb.set_trace()

            return torch.cat([x, pointfeat], 1), trans, trans_feat



class Pointnet_sem_seg(nn.Module):
    def __init__(self, k = 2, feature_transform=False,num_channel = 5):
        super(Pointnet_sem_seg, self).__init__()
        self.k = k
        if(num_channel<=3):
            feature_transform = False
        else:
            feature_transform = True
        self.feature_transform=feature_transform
        self.num_channel = num_channel
        self.feat = PointNetEncoder(global_feat=False, feature_transform=self.feature_transform, channel=self.num_channel )
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = x.transpose(2, 1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return [x, trans_feat]

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())


# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# from torch.autograd import Variable
# import numpy as np
# import torch.nn.functional as F
#
#
# class STN3d(nn.Module):
#     def __init__(self,num_channel=5):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(num_channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         #pdb.set_trace()
#         print(x.shape)
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
#             batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x
#
#
# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#         self.k = k
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x
#
# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k * k)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#         self.k = k
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
#             batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x
#
#
#
#
#
# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat = True, feature_transform = False):
#         super(PointNetfeat, self).__init__()
#         self.stn = STN3d()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)
#
#     def forward(self, x):
#         n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))
#
#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2,1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2,1)
#         else:
#             trans_feat = None
#
#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat
#
# class PointNetCls(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetCls, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), trans, trans_feat
#
#
# class PointNetDenseCls(nn.Module):
#     def __init__(self, k = 2, feature_transform=True,num_channel=5):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feat = PointNetEncoder(global_feat=False, feature_transform=True, num_channel=num_channel)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         # print(x.shape)
#         x, trans, trans_feat = self.feat(x)
#         # print(trans_feat.shape)
#
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans_feat
#
#
# class PointNetEncoder(nn.Module):
#     def __init__(self, global_feat=True, feature_transform=False, num_channel=5):
#         super(PointNetEncoder, self).__init__()
#         self.stn = STN3d(num_channel)
#         self.conv1 = torch.nn.Conv1d(num_channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)
#
#     def forward(self, x):
#         B, D, N = x.size()
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         if D >3 :
#             x, feature = x.split(3,dim=2)
#         x = torch.bmm(x, trans)
#         if D > 3:
#             x = torch.cat([x,feature],dim=2)
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))
#
#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2, 1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2, 1)
#         else:
#             trans_feat = None
#
#         #print(trans_feat.shape)
#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, N)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat
#
#
# def feature_transform_regularizer(trans):
#     d = trans.size()[1]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
#     return loss
#
# # def feature_transform_reguliarzer(trans):
# #     d = trans.size()[1]
# #     I = torch.eye(d)[None, :, :]
# #     if trans.is_cuda:
# #         I = I.cuda()
# #     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
# #     return loss
# #
# #
# # def feature_transform_regularizer(trans):
# #     d = trans.size()[1]
# #     batchsize = trans.size()[0]
# #     I = torch.eye(d)[None, :, :]
# #     if trans.is_cuda:
# #         I = I.cuda()
# #     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
# #     return loss
#
# if __name__ == '__main__':
#     sim_data = Variable(torch.rand(32,3,2500))
#     trans = STN3d()
#     out = trans(sim_data)
#     print('stn', out.size())
#     print('loss', feature_transform_regularizer(out))
#
#     sim_data_64d = Variable(torch.rand(32, 64, 2500))
#     trans = STNkd(k=64)
#     out = trans(sim_data_64d)
#     print('stn64d', out.size())
#     print('loss', feature_transform_regularizer(out))
#
#     pointfeat = PointNetfeat(global_feat=True)
#     out, _, _ = pointfeat(sim_data)
#     print('global feat', out.size())
#
#     pointfeat = PointNetfeat(global_feat=False)
#     out, _, _ = pointfeat(sim_data)
#     print('point feat', out.size())
#
#     cls = PointNetCls(k = 5)
#     out, _, _ = cls(sim_data)
#     print('class', out.size())
#
#     seg = PointNetDenseCls(k = 3)
#     out, _, _ = seg(sim_data)
#     print('seg', out.size())
