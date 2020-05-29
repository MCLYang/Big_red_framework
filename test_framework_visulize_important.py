from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('/')
from argparse import ArgumentParser
import os
import h5py
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
import time
import wandb
from collections import OrderedDict
import random
from BigredDataSet import BigredDataSet
from kornia.utils.metrics import mean_iou,confusion_matrix
import pandas as pd
import importlib
# import ckpt

# importlib.import_module
# MODEL = importlib.import_module(args.model)
# shutil.copy('models/%s.py' % args.model, str(experiment_dir))
# shutil.copy('models/pointnet_util.py', str(experiment_dir))


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

def visualize_wandb(points,pred,target,index_important):
    # points [B,N,C]->[B*N,C]
    # pred,target [B,N,1]->[B*N,1]
    points = points.view(-1,5).numpy()
    pred = pred.view(-1,1).numpy()
    target = target.view(-1,1).numpy()
    index_important = index_important.view(-1,)
    temp_arr = np.zeros(len(target))
    temp_arr[index_important] = 1
    temp_arr = temp_arr.reshape(-1,1)
    

    points_gt =np.concatenate((points[:,[0,1,2]],target),axis=1)
    points_pd =np.concatenate((points[:,[0,1,2]],pred),axis=1)
    points_important =np.concatenate((points[:,[0,1,2]],temp_arr),axis=1)


    wandb.log({"Ground_truth": wandb.Object3D(points_gt)})
    wandb.log({"Prediction": wandb.Object3D(points_pd)})
    wandb.log({"important points": wandb.Object3D(points_important)})



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


def opt_global_inti():
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='some_name')
    parser.add_argument('--notification_email', type=str, default='will@email.com')
    parser.add_argument('--num_gpu', type=int,default=1 ,help="num_gpu")
    parser.add_argument('--dataset_root', type=str, default='../bigRed_h5_pointnet_sorted', help="dataset path")
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=32)
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

    parser.add_argument('--phase', type=str,default='test' ,help="root load_pretrain")
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    parser.add_argument('--num_channel', type=int,default=5,help="num_channel")
    parser.add_argument('--num_points', type=int,default=20000 ,help="use feature transform")
    parser.add_argument('--debug', type=bool,default=False ,help="is task for debugging?False for load entire dataset")

    parser.add_argument('--load_pretrain', type=str,default='ckpt/pointnet_5c_simple',help="root load_pretrain")
    parser.add_argument('--model', type=str,default='Pointnet_ring_light' ,help="[pointnet,pointnetpp,deepgcn,dgcnn]")
    parser.add_argument('--including_ring', type=lambda x: (str(x).lower() == 'true'),default=False ,help="is task for debugging?False for load entire dataset")

    args = parser.parse_args()
    return args

def save_model(package,root):
    torch.save(package,root)

def generate_report(summery_dict,package):
    save_sheet=[]
    save_sheet.append(['name',package['name']])
    save_sheet.append(['validation_miou',package['Miou_validation_ave']])
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





def main():
    setSeed(10)
    opt = opt_global_inti()
    print('----------------------Load ckpt----------------------')
    pretrained_model_path = os.path.join(opt.load_pretrain,'best_model.pth')
    package = torch.load(pretrained_model_path)
    para_state_dict = package['state_dict']
    opt.num_channel = package['num_channel']
    opt.time = package['time'] 
    opt.epoch_ckpt = package['epoch']

    # opt.val_miou = package['validation_mIoU']
    # package.pop('validation_mIoU')
    # package['Validation_ave_miou'] = opt.val_miou

    # num_gpu = package['gpuNum']
    # package.pop('gpuNum')
    # package['num_gpu'] = num_gpu

    # save_model(package,pretrained_model_path)        
    state_dict = convert_state_dict(para_state_dict)

    ckpt_,ckpt_file_name  = opt.load_pretrain.split("/")
    module_name = ckpt_+'.'+ckpt_file_name+'.'+'model'
    MODEL = importlib.import_module(module_name)
    # print('opt.num_channel: ',opt.num_channel)
    model = MODEL.get_model(input_channel = opt.num_channel)
    Model_Specification = MODEL.get_model_name(input_channel = opt.num_channel)
    print('----------------------Test Model----------------------')
    print('Root of prestrain model: ', pretrained_model_path)
    print('Model: ', opt.model)
    print('Pretrained model name: ', Model_Specification)
    print('Trained Date: ',opt.time)
    print('num_channel: ',opt.num_channel)
    name = input("Edit the name or press ENTER to skip: ")
    if(name!=''):
        opt.model_name = name
    else:
        opt.model_name = Model_Specification
    print('Pretrained model name: ', opt.model_name)
    package['name'] = opt.model_name
    save_model(package,pretrained_model_path)       
    # pdb.set_trace() 
    # save_model(package,root,name)



    # if(model == 'pointnet'):
    #     #add args
    #     model = pointnet.Pointnet_sem_seg(k=2,num_channel=opt.num_channel)
    # elif(model == 'pointnetpp'):
    #     print()
    # elif(model == 'deepgcn'):
    #     print()
    # elif(model == 'dgcnn'):
    #     print()

    model.load_state_dict(state_dict)
    model.cuda()

    print('----------------------Load Dataset----------------------')
    print('Root of dataset: ', opt.dataset_root)
    print('Phase: ', opt.phase)
    print('debug: ', opt.debug)

    test_dataset = BigredDataSet(
        root=opt.dataset_root,
        is_train=False,
        is_validation=False,
        is_test=True,
        num_channel = opt.num_channel,
        test_code = opt.debug,
        including_ring = opt.including_ring)
    result_sheet = test_dataset.result_sheet
    file_dict= test_dataset.file_dict
    tag_Getter = tag_getter(file_dict)



    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=int(opt.num_workers))

    print('num_frame: ',len(test_dataset))
    print('batch_size: ', opt.batch_size)
    print('num_batch: ', int(len(testloader) / opt.batch_size))

    print('----------------------Testing----------------------')
    metrics_list = ['Miou','Biou','Fiou','test_loss','OA','time_complexicity','storage_complexicity']
    for name in result_sheet:
        metrics_list.append(name)

    manager = metrics_manager(metrics_list)

    model.eval()
    wandb.init(project="Test",name=package['name'])
    wandb.config.update(opt)

    points_gt_list =[]
    points_pd_list =[]
    points_important_list =[]

    with torch.no_grad():
        for j, data in tqdm(enumerate(testloader), total=len(testloader), smoothing=0.9):
            points, target = data
            #target.shape [B,N]
            #points.shape [B,N,C]
            points, target = points.cuda(), target.cuda()
            tic = time.perf_counter()
            pred_mics = model(points)
            toc = time.perf_counter()
            #pred_mics[0] is pred
            #pred_mics[1] is feat [only pointnet and pointnetpp has it]

            #compute loss
            test_loss = 0

            #pred.shape [B,N,2] since pred returned pass F.log_softmax
            pred, target,points = pred_mics[0].cpu(), target.cpu(),points.cpu()
            imp_glob = pred_mics[2].cpu()

            #pred:[B,N,2]->[B,N]
            # pdb.set_trace()
            pred = pred.data.max(dim=2)[1]
            #compute confusion matrix
            cm = confusion_matrix(pred,target,num_classes =2).sum(dim=0)
            #compute OA
            overall_correct_site = torch.diag(cm).sum()
            overall_reference_site = cm.sum()
            assert overall_reference_site == opt.batch_size * opt.num_points,"Confusion_matrix computing error" 
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
            manager.update('test_loss',test_loss)
            manager.update('OA',oa)
            manager.update('Biou',Biou.item())
            manager.update('Fiou',Fiou.item())
            manager.update('Miou',miou.item())
            manager.update('time_complexicity',float(1/time_complexity))
            manager.update('storage_complexicity',RAM_usagePeak.item())
            #get tags,compute the save miou for corresponding class
            difficulty,location,isSingle,file_name=tag_Getter.get_difficulty_location_isSingle(j)
            manager.update(file_name,miou.item())
            manager.update(difficulty,miou.item())
            manager.update(isSingle,miou.item())



            dim_num = points.shape[2]
            points = points.view(-1,dim_num).numpy()
            pred = pred.view(-1,1).numpy()
            target = target.view(-1,1).numpy()
            imp_glob = imp_glob.view(-1,)


            number_sheet,_,bin_sheet = torch.unique(imp_glob, sorted=True, return_inverse=True, return_counts=True, dim=None)

            temp_arr = np.zeros(len(target))

            temp_arr[number_sheet] = bin_sheet


            temp_arr = temp_arr.reshape(-1,1)
            points_gt =np.concatenate((points[:,[0,1,2]],target),axis=1)
            points_pd =np.concatenate((points[:,[0,1,2]],pred),axis=1)
            points_important =np.concatenate((points[:,[0,1,2]],temp_arr),axis=1)


            if(opt.including_ring):
                temp_arr2 = np.zeros(len(target))
                imp_ring = pred_mics[3].cpu()
                imp_ring = imp_ring.view(-1,)
                number_sheet,_,bin_sheet = torch.unique(imp_ring, sorted=True, return_inverse=True, return_counts=True, dim=None)
                temp_arr2[number_sheet] = bin_sheet
                temp_arr2 = temp_arr2.reshape(-1,1)
                points_important =np.concatenate((points_important,temp_arr2),axis=1)


            points_gt_list.append(points_gt)
            points_pd_list.append(points_pd)
            points_important_list.append(points_important)


            # visualize_wandb(points,pred,target,index_important)
            # pdb.set_trace()
    

    f = h5py.File('resluts.h5','w')
    f.create_dataset('points_gt_list',data = np.array(points_gt_list))
    f.create_dataset('points_pd_list',data =np.array(points_pd_list))
    f.create_dataset('points_important_list',data = np.array(points_important_list))
    f.close()



    summery_dict = manager.summary()
    generate_report(summery_dict,package)
    wandb.log(summery_dict)
    # wandb.save('model.h5')




if __name__ == '__main__':
    main()