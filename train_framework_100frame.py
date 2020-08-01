from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('/')
from argparse import ArgumentParser
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import numpy as np
import pdb
# from torch.utils.tensorboard import SummaryWriter
from glob import glob
import pandas as pd
from metrics_manager import metrics_manager
from pathlib import Path
import time
import wandb
from collections import OrderedDict
import random
from BigredDataSet_finetune100 import BigredDataSet  
from BigredDataSetPTG import BigredDataSetPTG
from kornia.utils.metrics import mean_iou,confusion_matrix
import pandas as pd
import importlib
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from opt_deepgcn import OptInit as OptInit_deepgcn
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms
from torch.nn import Sequential as Seq

# from apex import amp
# import apex


# import ckpt

# importlib.import_module
# MODEL = importlib.import_module(args.model)
# shutil.copy('models/%s.py' % args.model, str(experiment_dir))
# shutil.copy('models/pointnet_util.py', str(experiment_dir))



def opt_global_inti():
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='some_name')
    parser.add_argument('--notification_email', type=str, default='will@email.com')
    # parser.add_argument('--dataset_root', type=str, default='../bigRed_h5_pointnet', help="dataset path")
    # parser.add_argument('--dataset_root', type=str, default='../bigRed_h5_pointnet_sorted', help="dataset path")
    parser.add_argument('--dataset_root', type=str, default='../bigRed_h5_pointnet_sorted', help="dataset path")

    parser.add_argument('--apex', type=lambda x: (str(x).lower() == 'true'),default=False ,help="is task for debugging?False for load entire dataset")
    parser.add_argument('--opt_level', default='O2',type=str, metavar='N')

    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=32)

    parser.add_argument('--phase', type=str,default='Train' ,help="root load_pretrain")
    parser.add_argument('--num_points', type=int,default=20000 ,help="use feature transform")

    parser.add_argument('--wandb_history', type=lambda x: (str(x).lower() == 'true'),default=False ,help="load wandb history")
    parser.add_argument('--wandb_id', type=str,default='',help="")
    parser.add_argument('--wandb_file', type=str,default='',help="")

    parser.add_argument('--unsave_epoch', type=int,default=0,help="")
    parser.add_argument('--load_pretrain', type=str,default='',help="root load_pretrain")
    parser.add_argument('--synchonization', type=str,default='BN' ,help="[BN,BN_syn,Instance]")
    parser.add_argument('--tol_stop', type=float,default=1e-5 ,help="early stop for loss")
    parser.add_argument('--epoch_max', type=int,default=500,help="epoch_max")
    # parser.add_argument('--wd_project', type=str,default="Test_TimeComplexcity",help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")

    #Pointnet_ring_light4c_upsample+groupConv
    parser.add_argument('--num_gpu', type=int,default=2,help="num_gpu")

    parser.add_argument('--num_channel', type=int,default=4,help="num_channel")
    parser.add_argument('--model', type=str,default='pointnet' ,help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")
    # parser.add_argument('--model', type=str,default='pointnet' ,help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")

    parser.add_argument('--including_ring', type=lambda x: (str(x).lower() == 'true'),default=False ,help="is task for debugging?False for load entire dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument('--wd_project', type=str,default="100",help="")
    # parser.add_argument('--wd_project', type=str,default="debug",help="[pointnet,pointnetpp,deepgcn,dgcnn,pointnet_ring,pointnet_ring_light]")
    parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'),default=False ,help="is task for debugging?False for load entire dataset")

    args = parser.parse_args()
    return args

def save_model(package,root):
    torch.save(package,root)

def setSeed(seed = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def convert_state_dict(state_dict):
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def visualize_wandb(points,pred,target):
    # points [B,N,C]->[B*N,C]
    # pred,target [B,N,1]->[B*N,1]
    points = points.view(-1,5).numpy()
    pred = pred.view(-1,1).numpy()
    target = target.view(-1,1).numpy()
    points_gt =np.concatenate((points[:,[0,1,2]],target),axis=1)
    points_pd =np.concatenate((points[:,[0,1,2]],pred),axis=1)
    wandb.log({"Ground_truth": wandb.Object3D(points_gt)})
    wandb.log({"Prediction": wandb.Object3D(points_pd)})


class tag_getter(object):
    def __init__(self,file_dict):
        self.sorted_keys = np.array(sorted(file_dict.keys()))
        self.file_dict = file_dict
    def get_difficulty_location_isSingle(self,j):
        temp_arr = self.sorted_keys<=j
        index_for_keys = sum(temp_arr)
        _key = self.sorted_keys[index_for_keys-1]
        file_name = self.file_dict[_key]
        file_name = file_name[:-3]
        difficulty,location,isSingle = file_name.split("_")
        return(difficulty,location,isSingle,file_name)


def generate_report(summery_dict,package):
    save_sheet=[]
    save_sheet.append(['name',package['name']])
    save_sheet.append(['validation_miou',package['Validation_ave_miou']])
    save_sheet.append(['test_miou',summery_dict['Miou']])
    save_sheet.append(['Biou',summery_dict['Biou']])
    save_sheet.append(['Fiou',summery_dict['Fiou']])
    save_sheet.append(['time_complexicity(f/s)',summery_dict['time_complexicity']])
    save_sheet.append(['storage_complexicity',summery_dict['storage_complexicity']])
    save_sheet.append(['number_channel',package['num_channel']])
    save_sheet.append(['Date',package['time']])
    save_sheet.append(['Training-Validation-Testing','0.7-0.9-1'])
    
    for name in summery_dict:
        if(name!='Miou' 
            and name!='storage_complexicity'
            and name!='time_complexicity'
            and name!='Biou'
            and name!='Fiou'
            ):
            save_sheet.append([name,summery_dict[name]])
        print(name+': %2f' % summery_dict[name])
    # pdb.set_trace()
    save_sheet.append(['para',''])
    
    f = pd.DataFrame(save_sheet)
    f.to_csv('testReport.csv',index=False,header=None)


def load_pretrained(opt):
    print('---------------------load_pretrained----------------------')
    pretrained_model_path = os.path.join(opt.load_pretrain,'val_miou0.8188_Epoch119.pth')
    package = torch.load(pretrained_model_path)
    para_state_dict = package['state_dict']
    opt.num_channel = package['num_channel']
    opt.time = package['time'] 
    opt.epoch_ckpt = package['epoch']
    opt.val_miou = package['Miou_validation_ave']

    scheduler = package['scheduler']

    state_dict = convert_state_dict(para_state_dict)
    ckpt_,ckpt_file_name  = opt.load_pretrain.split("/")
    module_name = ckpt_+'.'+ckpt_file_name+'.'+'model'
    MODEL = importlib.import_module(module_name)
    model = MODEL.get_model(input_channel = opt.num_channel)
    model.load_state_dict(state_dict)


    Model_Specification = MODEL.get_model_name(input_channel = opt.num_channel)
    f_loss = MODEL.get_loss(input_channel = opt.num_channel)
    opt.model_name = package['model_name']


    print('----------------------Model Info----------------------')
    print('Root of prestrain model: ', pretrained_model_path)
    print('Model: ', opt.model)
    print('Model Specification: ', Model_Specification)
    print('Trained Date:',opt.time)
    print('num_channel:',opt.num_channel)
    print('Model name:',opt.model_name)
    print('----------------------Configure optimizer and scheduler----------------------')


    experiment_dir = Path('ckpt/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(opt.model_name)
    experiment_dir.mkdir(exist_ok=True)
    shutil.copy('model/%s.py' % opt.model, str(experiment_dir))
    shutil.move(os.path.join(str(experiment_dir), '%s.py'% opt.model), 
                os.path.join(str(experiment_dir), 'model.py'))
    experiment_dir = experiment_dir.joinpath('saves')
    experiment_dir.mkdir(exist_ok=True)
    opt.save_root = str(experiment_dir)

    if(opt.apex==True):
        # model = apex.parallel.convert_syncbn_model(model)
        model.cuda()
        f_loss.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model = torch.nn.DataParallel(model,device_ids =[0,1])
    else:
        # model = apex.parallel.convert_syncbn_model(model)
        model.cuda()
        f_loss.cuda()
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizer_dict = package['optimizer'].state_dict()
    optimizer.load_state_dict(optimizer_dict)


    return opt,model,f_loss,optimizer,scheduler


def creating_new_model(opt):
    print('----------------------Creating model----------------------')
    opt.time = time.ctime()
    opt.epoch_ckpt = 0
    opt.val_miou = 0
    module_name = 'model.'+opt.model
    MODEL = importlib.import_module(module_name)
    opt_deepgcn = None

    if(opt.model == 'deepgcn'):
        opt_deepgcn = OptInit_deepgcn().initialize()
        model = MODEL.get_model(opt2 = opt_deepgcn,input_channel = opt.num_channel,is_synchoization = opt.synchonization)
    else:
        model = MODEL.get_model(input_channel = opt.num_channel,is_synchoization = opt.synchonization)

    Model_Specification = MODEL.get_model_name(input_channel = opt.num_channel)
    f_loss = MODEL.get_loss(input_channel = opt.num_channel)

    print('----------------------Model Info----------------------')
    print('Root of prestrain model: ', '[No Prestrained loaded]')
    print('Model: ', opt.model)
    print('Model Specification: ', Model_Specification)
    print('Trained Date: ',opt.time)
    print('num_channel: ',opt.num_channel)
    name = input("Edit the name or press ENTER to skip: ")
    if(name!=''):
        opt.model_name = name
    else:
        opt.model_name = Model_Specification
    print('Model name: ', opt.model_name)

    print('----------------------Configure optimizer and scheduler----------------------')
    experiment_dir = Path('ckpt/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(opt.model_name)
    experiment_dir.mkdir(exist_ok=True)
    shutil.copy('model/%s.py' % opt.model, str(experiment_dir))
    shutil.move(os.path.join(str(experiment_dir), '%s.py'% opt.model), 
                os.path.join(str(experiment_dir), 'model.py'))
    experiment_dir = experiment_dir.joinpath('saves')
    experiment_dir.mkdir(exist_ok=True)
    opt.save_root = str(experiment_dir)

    if(opt.apex==True):
        # model = apex.parallel.convert_syncbn_model(model)
        model.cuda()
        f_loss.cuda()

        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model = torch.nn.DataParallel(model,device_ids =[0,1])
    else:
        # model = apex.parallel.convert_syncbn_model(model)
        model.cuda()
        f_loss.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        model = torch.nn.DataParallel(model)

    return opt,model,f_loss,optimizer,scheduler,opt_deepgcn



# def creating_new_model(point_set,label_set):
#     #point_set [N,M,D]






def main():
    setSeed(10)
    opt = opt_global_inti()

    num_gpu = torch.cuda.device_count()
    assert num_gpu == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 

    gpu_name_list = []
    for i in range(num_gpu):
        gpu_name_list.append(torch.cuda.get_device_name(i))

    opt.gpu_list = gpu_name_list

    if(opt.load_pretrain!=''):
        opt,model,f_loss,optimizer,scheduler,opt_deepgcn = load_pretrained(opt)
    else:
        opt,model,f_loss,optimizer,scheduler,opt_deepgcn = creating_new_model(opt)
    


    print('----------------------Load Dataset----------------------')
    print('Root of dataset: ', opt.dataset_root)
    print('Phase: ', opt.phase)
    print('debug: ', opt.debug)

    #pdb.set_trace()

    if(opt.model!='deepgcn'):
        train_dataset = BigredDataSet(
            root=opt.dataset_root,
            is_train=True,
            is_validation=False,
            is_test=False,
            num_channel = opt.num_channel,
            test_code = opt.debug,
            including_ring = opt.including_ring
            )

        f_loss.load_weight(train_dataset.labelweights)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=int(opt.num_workers))

        validation_dataset = BigredDataSet(
            root=opt.dataset_root,
            is_train=False,
            is_validation=True,
            is_test=False,
            num_channel = opt.num_channel,
            test_code = opt.debug,
            including_ring = opt.including_ring)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=int(opt.num_workers))
    else:
        train_dataset = BigredDataSetPTG(root = opt.dataset_root,
                                 is_train=True,
                                 is_validation=False,
                                 is_test=False,
                                 num_channel=opt.num_channel,
                                 new_dataset = False,
                                 test_code = opt.debug,
                                 pre_transform=torch_geometric.transforms.NormalizeScale()
                                 )
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        validation_dataset = BigredDataSetPTG(root = opt.dataset_root,
                                    is_train=False,
                                    is_validation=True,
                                    is_test=False,
                                    new_dataset = False,
                                    test_code = opt.debug,
                                    num_channel=opt.num_channel,
                                    pre_transform=torch_geometric.transforms.NormalizeScale()
                                    )
        validation_loader = DenseDataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

        labelweights = np.zeros(2)
        labelweights, _ = np.histogram(train_dataset.data.y.numpy(), range(3))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        weights = torch.Tensor(labelweights).cuda()
        f_loss.load_weight(weights)







    print('train dataset num_frame: ',len(train_dataset))
    print('num_batch: ', int(len(train_loader) / opt.batch_size))


    print('validation dataset num_frame: ',len(validation_dataset))
    print('num_batch: ', int(len(validation_loader) / opt.batch_size))

    print('Batch_size: ', opt.batch_size)

    print('----------------------Prepareing Training----------------------')
    metrics_list = ['Miou','Biou','Fiou','loss','OA','time_complexicity','storage_complexicity']
    manager_test = metrics_manager(metrics_list)

    metrics_list_train = ['Miou','Biou',
                            'Fiou','loss',
                            'storage_complexicity',
                            'time_complexicity']
    manager_train = metrics_manager(metrics_list_train)


    wandb.init(project=opt.wd_project,name=opt.model_name,resume=False)
    if(opt.wandb_history == False):
        best_value = 0
    else:
        temp = wandb.restore('best_model.pth',run_path = opt.wandb_id)
        best_value = torch.load(temp.name)['Miou_validation_ave']

    wandb.config.update(opt)

    if opt.epoch_ckpt == 0:
        opt.unsave_epoch = 0
    else:
        opt.epoch_ckpt = opt.epoch_ckpt+1

    for epoch in range(opt.epoch_ckpt,opt.epoch_max):
        manager_train.reset()
        model.train()
        tic_epoch = time.perf_counter()
        print('---------------------Training----------------------')
        print("Epoch: ",epoch)
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            
            if(opt.model == 'deepgcn'):
                points = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
                points = points[:, :opt.num_channel, :, :]
                target = data.y.cuda()
            else:
                points, target = data
                #target.shape [B,N]
                #points.shape [B,N,C]
                points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # pdb.set_trace()
            #training...
            optimizer.zero_grad()
            tic = time.perf_counter()
            pred_mics = model(points)                
            toc = time.perf_counter()
            #compute loss

            #For loss
            #target.shape [B,N] ->[B*N]
            #pred.shape [B,N,2]->[B*N,2]
            #pdb.set_trace()
            

            #pdb.set_trace()
            loss = f_loss(pred_mics, target)   

            if(opt.apex):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            #pred.shape [B,N,2] since pred returned pass F.log_softmax
            pred, target = pred_mics[0].cpu(), target.cpu()

            #pred:[B,N,2]->[B,N]
            #pdb.set_trace()
            pred = pred.data.max(dim=2)[1]
            
            #compute iou
            Biou,Fiou = mean_iou(pred,target,num_classes =2).mean(dim=0)
            miou = (Biou+Fiou)/2

            #compute Training time complexity
            time_complexity = toc - tic


            #compute Training storage complexsity
            num_device = torch.cuda.device_count()
            assert num_device == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 
            temp = []
            for k in range(num_device):
                temp.append(torch.cuda.memory_allocated(k))
            RAM_usagePeak = torch.tensor(temp).float().mean()

            #print(loss.item())
            #print(miou.item())
            #writeup logger
            manager_train.update('loss',loss.item())
            manager_train.update('Biou',Biou.item())
            manager_train.update('Fiou',Fiou.item())
            manager_train.update('Miou',miou.item())
            manager_train.update('time_complexicity',float(1/time_complexity))
            manager_train.update('storage_complexicity',RAM_usagePeak.item())

            log_dict = {'loss_online':loss.item(),
                        'Biou_online':Biou.item(),
                        'Fiou_online':Fiou.item(),
                        'Miou_online':miou.item(),
                        'time_complexicity_online':float(1/time_complexity),
                        'storage_complexicity_online':RAM_usagePeak.item()
                        }
            if(epoch - opt.unsave_epoch>=0):
                wandb.log(log_dict)

        toc_epoch = time.perf_counter()
        time_tensor = toc_epoch-tic_epoch


        summery_dict = manager_train.summary()
        log_train_end = {}
        for key in summery_dict:
            log_train_end[key+'_train_ave'] = summery_dict[key]
            print(key+'_train_ave: ',summery_dict[key])
        
        log_train_end['Time_PerEpoch'] = time_tensor
        if(epoch - opt.unsave_epoch>=0):
            wandb.log(log_train_end)
        else:
            print('No data upload to wandb. Start upload: Epoch[%d] Current: Epoch[%d]'%(opt.unsave_epoch,epoch))

        scheduler.step()
        if(epoch % 10 == 1):
            print('---------------------Validation----------------------')
            manager_test.reset()
            model.eval()
            print("Epoch: ",epoch)
            with torch.no_grad():
                for j, data in tqdm(enumerate(validation_loader), total=len(validation_loader), smoothing=0.9):


                    if(opt.model == 'deepgcn'):
                        points = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
                        points = points[:, :opt.num_channel, :, :]
                        target = data.y.cuda()
                    else:
                        points, target = data
                        #target.shape [B,N]
                        #points.shape [B,N,C]
                        points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)


                    tic = time.perf_counter()
                    pred_mics = model(points)                
                    toc = time.perf_counter()
                    
                    #pred.shape [B,N,2] since pred returned pass F.log_softmax
                    pred, target = pred_mics[0].cpu(), target.cpu()

                    #compute loss
                    test_loss = 0

                    #pred:[B,N,2]->[B,N]
                    pred = pred.data.max(dim=2)[1]
                    #compute confusion matrix
                    cm = confusion_matrix(pred,target,num_classes =2).sum(dim=0)
                    #compute OA
                    overall_correct_site = torch.diag(cm).sum()
                    overall_reference_site = cm.sum()
                    # if(overall_reference_site != opt.batch_size * opt.num_points):
                    #pdb.set_trace()
                    #assert overall_reference_site == opt.batch_size * opt.num_points,"Confusion_matrix computing error"
                    oa = float(overall_correct_site/overall_reference_site)
                    
                    #compute iou
                    Biou,Fiou = mean_iou(pred,target,num_classes =2).mean(dim=0)
                    miou = (Biou+Fiou)/2

                    #compute inference time complexity
                    time_complexity = toc - tic
                    
                    #compute inference storage complexsity
                    num_device = torch.cuda.device_count()
                    assert num_device == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 
                    temp = []
                    for k in range(num_device):
                        temp.append(torch.cuda.memory_allocated(k))
                    RAM_usagePeak = torch.tensor(temp).float().mean()
                    #writeup logger
                    # metrics_list = ['test_loss','OA','Biou','Fiou','Miou','time_complexicity','storage_complexicity']
                    manager_test.update('loss',test_loss)
                    manager_test.update('OA',oa)
                    manager_test.update('Biou',Biou.item())
                    manager_test.update('Fiou',Fiou.item())
                    manager_test.update('Miou',miou.item())
                    manager_test.update('time_complexicity',float(1/time_complexity))
                    manager_test.update('storage_complexicity',RAM_usagePeak.item())

            
            summery_dict = manager_test.summary()

            log_val_end = {}
            for key in summery_dict:
                log_val_end[key+'_validation_ave'] = summery_dict[key]
                print(key+'_validation_ave: ',summery_dict[key])

            package = dict()
            package['state_dict'] = model.state_dict()
            package['scheduler'] = scheduler
            package['optimizer'] = optimizer
            package['epoch'] = epoch

            opt_temp = vars(opt)
            for k in opt_temp:
                package[k] = opt_temp[k]
            if(opt_deepgcn is not None):
                opt_temp = vars(opt_deepgcn)
                for k in opt_temp:
                    package[k+'_opt2'] = opt_temp[k]


            for k in log_val_end:
                package[k] = log_val_end[k]

            save_root = opt.save_root+'/val_miou%.4f_Epoch%s.pth'%(package['Miou_validation_ave'],package['epoch'])
            torch.save(package,save_root)

            print('Is Best?: ',(package['Miou_validation_ave']>best_value))
            if(package['Miou_validation_ave']>best_value):
                best_value = package['Miou_validation_ave']
                save_root = opt.save_root+'/best_model.pth'
                torch.save(package,save_root)
            if(epoch - opt.unsave_epoch>=0):
                wandb.log(log_val_end)
            else:
                print('No data upload to wandb. Start upload: Epoch[%d] Current: Epoch[%d]'%(opt.unsave_epoch,epoch))
            if(opt.debug == True):
                pdb.set_trace()




if __name__ == '__main__':
    main()
