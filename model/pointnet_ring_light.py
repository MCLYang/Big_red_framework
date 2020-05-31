from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
APEX_AVAILABLE = True


def get_model(input_channel,is_synchoization='BN'):
    return(Pointnet_ring_light_sem_seg(num_channel=input_channel,is_synchoization = is_synchoization))

def get_model_name(input_channel):
    return("Pointnet_ring_light"+str(input_channel)+"c")

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
        target = target.view(-1)
        pred = pred_mics[0].cpu()
        trans_feat = pred_mics[1]
        pred = pred.view(-1,2)
        # pdb.set_trace()
        if(self.feature_transform == True):
            loss = F.nll_loss(pred, target, weight = weight)
            mat_diff_loss = feature_transform_regularizer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        else:
            total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss



class STN3d(nn.Module):
    def __init__(self,num_channal = 5,is_synchoization = 'BN'):
        super(STN3d, self).__init__()
        self.num_channal = num_channal
        self.conv1 = torch.nn.Conv1d(num_channal, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.k = 3
        if(is_synchoization == 'BN'):
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        else:
            self.bn1 = nn.InstanceNorm1d(64)
            self.bn2 = nn.InstanceNorm1d(128)
            self.bn3 = nn.InstanceNorm1d(1024)
            self.bn4 = nn.InstanceNorm1d(512)
            self.bn5 = nn.InstanceNorm1d(256)


    def forward(self, x):

        batchsize = x.size()[0]
        x = self.conv1(x)
        #print(x.dtype)
        x = self.bn1(x)
        #print(x.dtype)

        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        #print(x.shape)#[6, 1024, 1]
        # x = x.view(batchsize,1, 1024)
        x = x.view(-1, 1024)
        #pdb.set_trace()
        x = F.relu(self.bn4(self.fc1(x)))
        #pdb.set_trace()
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        #print('x',x.shape)
        x = x.squeeze(1)
        #print(x.dtype)
        if(APEX_AVAILABLE):
            iden = Variable(torch.eye(self.k,dtype = torch.float16).flatten()).view(1,self.k*self.k).repeat(batchsize,1)
        else:
            iden = Variable(torch.eye(self.k,dtype = torch.float32).flatten()).view(1,self.k*self.k).repeat(batchsize,1)
        
        #print(iden.dtype)
        #pdb.set_trace()

        #print('iden',iden.shape)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        #print(x.shape)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64,is_synchoization = 'BN'):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        if(is_synchoization == 'BN'):
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        else:
            self.bn1 = nn.InstanceNorm1d(64)
            self.bn2 = nn.InstanceNorm1d(128)
            self.bn3 = nn.InstanceNorm1d(1024)
            self.bn4 = nn.InstanceNorm1d(512)
            self.bn5 = nn.InstanceNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(batchsize,1, 1024)

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(1)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)

        

        if(APEX_AVAILABLE):
            iden = Variable(torch.eye(self.k,dtype = torch.float16).flatten()).view(1,self.k*self.k).repeat(batchsize,1)
        else:
            iden = Variable(torch.eye(self.k,dtype = torch.float32).flatten()).view(1,self.k*self.k).repeat(batchsize,1)


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

        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        # print(x.shape)
        # print(trans.shape)
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
        
        
        
        #print(trans_feat.shape)
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class maxpooler_global(nn.Module):
    def __init__(self,is_synchoization='BN'):
        super(maxpooler_global, self).__init__()
        if(is_synchoization=='BN'):
            self.pooler = nn.Sequential(
                    nn.Conv1d(64, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 1024, 1),
                    nn.BatchNorm1d(1024))
        else:
            self.pooler = nn.Sequential(
                    nn.Conv1d(64, 128, 1),
                    nn.InstanceNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 1024, 1),
                    nn.InstanceNorm1d(1024))


        
    def forward(self,x):
        #[B,D,N]
        N = x.size()[2]
        x = self.pooler(x)
        #pdb.set_trace()
        # x torch.Size([B, 1024, N]) -> torch.Size([B, 1024, 1])
        x = torch.max(x, 2, keepdim=True)[0]
        # x torch.Size([B, 1024, 1] -> torch.Size([B, 1024])
        x = x.view(-1, 1024)
        # torch.Size([B, 1024]) -> torch.Size([B, 1024, N])
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        return x




class maxpooler_ring(nn.Module):
    def __init__(self,num_ring=16,is_synchoization='BN'):
        super(maxpooler_ring, self).__init__()
        self.num_ring = num_ring        
        if(is_synchoization=='BN'):
            self.pooler = nn.Sequential(nn.Conv1d(64, 128, 1,groups=num_ring),nn.BatchNorm1d(128))
        else:
            self.pooler = nn.Sequential(nn.Conv1d(64, 128, 1,groups=num_ring),nn.InstanceNorm1d(128))

    def forward(self,x,ring):
        x = x.transpose(2, 1)
        # ring [B,N]
        # x [B,N,D]
        B,N,D = x.size()
        max_group = 0
        max_ring = 1520
        new_x = []
        for b in range(B):
            x_b = x[b,:,:]
            ring_b = ring[b,:]
            #find the endpoint for each ring
            if(APEX_AVAILABLE):
                number_sheet,_,bin_sheet = torch.unique(ring_b.to(torch.float16), sorted=True, return_inverse=True, return_counts=True, dim=None)
            else:
                number_sheet,_,bin_sheet = torch.unique(ring_b.to(torch.float32), sorted=True, return_inverse=True, return_counts=True, dim=None)
            ring_endpoint_dict = {}
            str_point_idx = 0
            for k in range(len(number_sheet)):
                if(k==0):
                    str_point_idx = torch.tensor(0).cuda()
                else:
                    str_point_idx = end_point_idx
                end_point_idx = (bin_sheet[:k+1]).sum()
                ring_endpoint_dict[int(number_sheet[k])] = [str_point_idx,end_point_idx]
            new_xb = []
            for key in ring_endpoint_dict:
                #key is num_ring
                #xi is [m_i,D]
                startpoint_mi = ring_endpoint_dict[key][0]
                endpoint_mi  = ring_endpoint_dict[key][1]
                xi = x_b[startpoint_mi:endpoint_mi,:]
                upsamle_id = torch.randperm(len(xi))[:(max_ring-len(xi))]
                choice = torch.arange(len(xi))
                choice = torch.cat((choice,upsamle_id),dim=0)
                xi = xi[choice,:]
                new_xb.append(xi)
            new_xb = torch.cat(new_xb,dim=0)
            new_x.append(new_xb)
        # pdb.set_trace()
        x = torch.cat(new_x,dim=0).view(B,64,-1)

        #BDN->B128N
        x = self.pooler(x)


        # pdb.set_trace()
        # for key in ring_endpoint_dict:
        x_new = []

        for i in range(16):
            x_temp = x[:,:,max_ring*i:max_ring*(i+1)]
            # x torch.Size([B, 128, N]) -> torch.Size([B, 128, 1])
            x_temp = torch.max(x_temp, 2, keepdim=True)[0]
            # x torch.Size([B, 128, 1] -> torch.Size([B, 128])
            x_temp = x_temp.view(-1, 128)
            startpoint_mi = ring_endpoint_dict[i][0]
            endpoint_mi  = ring_endpoint_dict[i][1]
            # torch.Size([B, 128]) -> torch.Size([B, 128, N])
            x_temp = x_temp.view(-1, 128, 1).repeat(1, 1, endpoint_mi -startpoint_mi)
            x_new.append(x_temp)

        # pdb.set_trace()
        x_new = torch.cat(x_new,dim=2)
        return x_new        

class RingEncoder(nn.Module):
    def __init__(self, global_feat=False, feature_transform=True, channel=5,num_ring = 16,is_synchoization='BN'):
        super(RingEncoder, self).__init__()
        self.channel = channel
        self.stn = STN3d(channel,is_synchoization = is_synchoization)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)

        if(is_synchoization == 'BN'):
            self.bn1 = nn.BatchNorm1d(64)
        else:
            self.bn1 = nn.InstanceNorm1d(64)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64,is_synchoization = is_synchoization)

        self.globalmaxpooler = maxpooler_global(is_synchoization = is_synchoization)
        self.ringmaxpooler = maxpooler_ring(num_ring=num_ring,is_synchoization = is_synchoization)
        

    def forward(self, x,ring):

        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        # print(x.shape)
        # print(trans.shape)
        # print('x.dtype',x.dtype)
        # print('trans.dtype',trans.dtype)
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

        x1 = self.globalmaxpooler(x)
        x2 = self.ringmaxpooler(x,ring)
        # return torch.cat([x1,x], 1), trans, trans_feat
        #pdb.set_trace()
        return torch.cat([x1,x2, x], 1), trans, trans_feat




class Pointnet_ring_light_sem_seg(nn.Module):
    def __init__(self, k = 2, feature_transform=True,num_channel = 4,num_ring=16,is_synchoization = 'BN'):
        super(Pointnet_ring_light_sem_seg, self).__init__()
        self.k = k
        feature_transform = True
        self.num_ring = num_ring
        self.feature_transform=feature_transform
        self.num_channel = num_channel
        self.feat = RingEncoder(global_feat=False, feature_transform=self.feature_transform, channel=self.num_channel,num_ring = 16,is_synchoization=is_synchoization)

        # self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv1 = torch.nn.Conv1d(1088+128, 512, 1)

        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        if(is_synchoization == 'BN'):
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)
        else:
            self.bn1 = nn.InstanceNorm1d(512)
            self.bn2 = nn.InstanceNorm1d(256)
            self.bn3 = nn.InstanceNorm1d(128)
    def forward(self, x):
        #pdb.set_trace()
        ring  = x[:,:,self.num_channel]
        x = x[:,:,:self.num_channel]

        x = x.transpose(2, 1)
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x, trans, trans_feat = self.feat(x,ring)
        #x is [B,N,1088] is pointnetEncoder
        #x is [B,N,1088+1024] is RingEncoder

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat

def feature_transform_regularizer(trans):
    #pdb.set_trace()
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    product =  torch.bmm(trans, trans.transpose(2,1)) - I
    # product = product.cpu()
    loss = torch.mean(torch.norm(product, dim=(1,2)))
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


