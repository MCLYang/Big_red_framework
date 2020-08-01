import os
import os.path as osp
import shutil
import numpy as np
import h5py
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
import shutil                                 

class BigredDataSetPTG(InMemoryDataset):
    def __init__(self,
                 root,
                 is_train=True,
                 is_validation=False,
                 is_test=False,
                 test_code = False,
                 new_dataset = True,
                 num_channel=5,
                 transform = None,
                 pre_transform=None,
                 pre_filter=None):
        self.is_train = is_train
        self.is_validation = is_validation
        self.is_test = is_test
        self.num_channel = num_channel
        self.test_code = test_code
        self.file_dict = {}
        self.result_sheet = []
        self.root = root

        if(is_train==True and is_validation==False and is_test==False):
            phase = 'train'
        elif(is_train==False and is_validation==True and is_test==False):
            phase = 'validation'
        elif(is_train==False and is_validation==False and is_test==True):
            phase = 'test'
        print("Loading dataset: ", phase)
        if(new_dataset == True):
            if(os.path.exists(os.path.join(root,"processed"))):
                print('Deleting the old data...')
                shutil.rmtree(os.path.join(root,"processed"))


        super(BigredDataSetPTG, self).__init__(root, transform, pre_transform, pre_filter)
        if(is_train==True and is_validation==False and is_test==False):
            path = self.processed_paths[0]
        elif(is_train==False and is_validation==True and is_test==False):
            path = self.processed_paths[1]
        elif(is_train==False and is_validation==False and is_test==True):
            path = self.processed_paths[2]
            #self.file_dict = torch.load(os.path.join(root,'file_dict.pth'))

        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        if(self.is_train==True and self.is_validation==False and self.is_test==False) or (self.is_train==False and self.is_validation==True and self.is_test==False):
            return ['train.txt']
        elif(self.is_train==False and self.is_validation==False and self.is_test==True):
            return ['test.txt']


    @property
    def processed_file_names(self):
        return ['{}.pt'.format(s) for s in ['train', 'evaluation','test']]

    # def download(self):
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.root)
    #     os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     name = self.url.split(os.sep)[-1].split('.')[0]
    #     os.rename(osp.join(self.root, name), self.raw_dir)
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]
            
            
        if(self.is_train==True or self.is_validation==True):
            xs_train, ys_train = [], []
            xs_validation, ys_validation = [], []
            file_dict ={}
            data_xyz = 0
            train_data_list,validation_data_list = [],[]
            counter_for_file = 0
            for filename in filenames:
                print('Processing: ',osp.join(self.raw_dir, filename))
                # f = h5py.File(osp.join(self.raw_dir, filename))
                with h5py.File(osp.join(self.raw_dir, filename),'r') as f:
                    data_xyz = f['xyz'][:]
                    if(self.test_code == False):
                        train_index = int(data_xyz.shape[0]*0.8)
                        validation_index = int(data_xyz.shape[0]*1)
                    else:
                        train_index = int(data_xyz.shape[0]*0.0008)
                        validation_index = int(data_xyz.shape[0]*0.001)


                    a = np.zeros((data_xyz.shape[0],data_xyz.shape[1],data_xyz.shape[2]+2))
                    a[:,:,:3] = f['xyz'][:]
                    a[:,:,3] = f['intensity'][:]
                    a[:,:,4] = f['laserID'][:]

                    a = a[:,:,:self.num_channel]
                    xs_train += torch.from_numpy(a[:train_index,:,:]).to(torch.float).unbind(0)
                    ys_train += torch.from_numpy(f['label'][:train_index]).to(torch.long).unbind(0)

                    xs_validation += torch.from_numpy(a[train_index:validation_index,:,:]).to(torch.float).unbind(0)
                    ys_validation += torch.from_numpy(f['label'][train_index:validation_index]).to(torch.long).unbind(0)
                    
            train_data_list,validation_data_list = [],[]
            for i, (x, y) in enumerate(zip(xs_train, ys_train)):
                data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                train_data_list.append(data)

            for i, (x, y) in enumerate(zip(xs_validation, ys_validation)):
                data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                validation_data_list.append(data)

            torch.save(self.collate(train_data_list), self.processed_paths[0])
            torch.save(self.collate(validation_data_list), self.processed_paths[1])
        else:
            xs_test, ys_test = [], []
            file_dict ={}
            data_xyz = 0
            train_data_list, test_data_list,validation_data_list = [],[],[]
            counter_for_file = 0
            for filename in filenames:
                print('Loading...',osp.join(self.raw_dir, filename))
                with h5py.File(osp.join(self.raw_dir, filename),'r') as f:
                    data_xyz = f['xyz'][:]
                    if(self.test_code == False):
                        test_index =  int(data_xyz.shape[0]*1)
                    else:
                        test_index =  int(data_xyz.shape[0]*0.01)


                    a = np.zeros((data_xyz.shape[0],data_xyz.shape[1],data_xyz.shape[2]+2))
                    a[:,:,:3] = f['xyz'][:]
                    a[:,:,3] = f['intensity'][:]
                    a[:,:,4] = f['laserID'][:]

                    a = a[:,:,:self.num_channel]

                    xs_test += torch.from_numpy(a[:test_index,:,:]).to(torch.float).unbind(0)
                    ys_test += torch.from_numpy(f['label'][:test_index]).to(torch.long).unbind(0)
                    
                    n_frame = np.array(f['xyz'][:test_index, :, :]).shape[0]
                    file_dict[counter_for_file] = filename
                    counter_for_file = counter_for_file + n_frame

            sorted_keys = np.array(sorted(file_dict.keys()))
            result_sheet = {
            'Complex':[],
            'Medium':[],
            'Simple':[],
            'multiPeople':[],
            'singlePerson':[]
            }
            for key in sorted_keys:
                tempname = file_dict[key]
                tempname = tempname[:-3]
                result_sheet[tempname] = []
            self.result_sheet = result_sheet
            self.file_dict = file_dict
            print(self.file_dict)
            torch.save(file_dict,os.path.join(self.root,'file_dict_test.pth'))
            test_data_list = []
            for i, (x, y) in enumerate(zip(xs_test, ys_test)):
                data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                test_data_list.append(data)
            torch.save(self.collate(test_data_list), self.processed_paths[2])

        
