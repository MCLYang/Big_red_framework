import sys
sys.path.append('model/')
import torch
import pdb
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq
import torch.nn as nn
import torch.nn.functional as F

def get_model(opt2,input_channel,is_synchoization = 'Instance'):
    return(deepgcn_sem_seg(opt=opt2,num_channel=input_channel,synchoization = is_synchoization))

def get_model_name(input_channel):
    return("deepgcn_"+str(input_channel)+"c")

def get_loss(input_channel):
    return(deepgcn_loss())


class deepgcn_loss(nn.Module):
    def __init__(self):
        super(deepgcn_loss, self).__init__()
        self.weight = torch.tensor([1,1]).cuda()
    def load_weight(self,weight):
        self.weight = torch.tensor(weight).cuda()
    def forward(self, pred_mics, target):
        weight = self.weight
        #pdb.set_trace()
        target = target.view(-1)
        pred = pred_mics[0]
        pred = pred.reshape(-1,2)
        loss = F.cross_entropy(pred, target,weight = weight, reduction='mean')
        return loss




class deepgcn_sem_seg(torch.nn.Module):
    def __init__(self,opt,num_channel = 4,synchoization='Instance'):
        super(deepgcn_sem_seg, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = BasicConv([channels+c_growth*(self.n_blocks-1), 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([channels+c_growth*(self.n_blocks-1)+1024, 512], act, norm, bias),
                                BasicConv([512, 256], act, norm, bias),
                                torch.nn.Dropout(p=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        B,C,N,_ = inputs.shape
        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        predict = self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)
        predict = predict.transpose(1,2)
        
        return [predict]
