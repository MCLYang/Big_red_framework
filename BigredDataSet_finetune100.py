import os
import os.path as osp
import shutil
import numpy as np
import h5py
import torch
import pdb
# from torch_geometric.data import (InMemoryDataset, Data, download_url,
#
#
#                                   extract_zip)

#is_test is final
import pdb
class BigredDataSet():
    def __init__(self,
                 root,
                 is_train=True,
                 is_validation=False,
                 is_test=False,
                 num_channel=5,
		         test_code = False,
                 including_ring = True
                 ):
        assert (num_channel >= 3), "num_channel must be equals or greater than 3. XYZ must be included!"

        # shape of data:1x,2y,3z,4ins,5laserID
        self.is_train = is_train
        self.root = root
        self.is_test = is_test
        self.is_validation = is_validation
        self.test_code = test_code
        self.file_dict = {}
        point_set = []
        label_set = []
        laserID_set = []
        intensity_set = []

        with open(os.path.join(root, "testframe.txt"), 'r') as f:
            data_list = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        # data_list = data_list[:1]
        pointset = []
        lableset = []
        counter_for_file = 0
        for file in data_list:
            # print(len(pointset))
            # pdb.set_trace()
            with h5py.File(os.path.join(root, file), 'r') as f:
                try:
                    print('Processing: ' + file)
                    if(self.test_code == False):
                            train_tail = int(100)
                            validation_tail = int(np.array(f['label']).shape[0] * 1)
                            test_tail = int(np.array(f['label']).shape[0] * 1)
                    if(self.test_code == True):
                            train_tail = int(np.array(f['label']).shape[0] * 0.01)
                            validation_tail = int(np.array(f['label']).shape[0] * 0.02)
                            test_tail = int(np.array(f['label']).shape[0] * 0.03)
                    current_point = []
                    if (self.is_train == True and self.is_validation == False and self.is_test == False):
                        print("Loading Training Data...")
                        n_frame = np.array(f['xyz'][:train_tail, :, :]).shape[0]
                        n_points = np.array(f['xyz'][:train_tail, :, :]).shape[1]
                        current_point.append(np.array(f['xyz'][:train_tail, :, :]))
                        current_point.append(np.array(f['intensity'][:train_tail, :]).reshape(n_frame, n_points, 1))
                        current_point.append(np.array(f['laserID'][:train_tail, :]).reshape(n_frame, n_points, 1))
                        current_point = np.concatenate(current_point, axis=2)
                        lableset.append(np.array(f['label'][:train_tail, :]))
                        pointset.append(current_point)

                        self.file_dict[counter_for_file] = file
                        counter_for_file = counter_for_file + n_frame

                    elif (self.is_train == False and self.is_validation == True and self.is_test == False):
                        print("Loading Validation Data...")
                        n_frame = np.array(f['xyz'][train_tail:validation_tail, :, :]).shape[0]
                        n_points = np.array(f['xyz'][train_tail:validation_tail, :, :]).shape[1]
                        current_point.append(np.array(f['xyz'][train_tail:validation_tail, :, :]))
                        current_point.append(
                            np.array(f['intensity'][train_tail:validation_tail, :]).reshape(n_frame, n_points, 1))
                        current_point.append(
                            np.array(f['laserID'][train_tail:validation_tail, :]).reshape(n_frame, n_points, 1))

                        current_point = np.concatenate(current_point, axis=2)
                        lableset.append(np.array(f['label'][train_tail:validation_tail, :]))
                        pointset.append(current_point)

                        self.file_dict[counter_for_file] = file
                        counter_for_file = counter_for_file + n_frame

                    elif (self.is_train == False and self.is_validation == False and self.is_test == True):
                        print("Loading Testing Data...")
                        n_frame = np.array(f['xyz'][validation_tail:test_tail, :, :]).shape[0]
                        n_points = np.array(f['xyz'][validation_tail:test_tail, :, :]).shape[1]
                        #print(n_points,n_frame)
                        current_point.append(np.array(f['xyz'][validation_tail:test_tail, :, :]))
                        current_point.append(
                            np.array(f['intensity'][validation_tail:test_tail, :]).reshape(n_frame, n_points, 1))
                        current_point.append(
                            np.array(f['laserID'][validation_tail:test_tail, :]).reshape(n_frame, n_points, 1))

                        current_point = np.concatenate(current_point, axis=2)
                        lableset.append(np.array(f['label'][validation_tail:test_tail, :]))
                        pointset.append(current_point)

                        self.file_dict[counter_for_file] = file
                        counter_for_file = counter_for_file + n_frame



                except:
                    f.close()

        sorted_keys = np.array(sorted(self.file_dict.keys()))

        result_sheet = {
        'Complex':[],
        'Medium':[],
        'Simple':[],
        'multiPeople':[],
        'singlePerson':[]
        }

        for key in sorted_keys:
            tempname = self.file_dict[key]
            tempname = tempname[:-3]
            result_sheet[tempname] = []
        self.result_sheet = result_sheet


        #pdb.set_trace()
        # pdb.set_trace()
        self.point_set = np.concatenate(pointset, axis=0)
        self.label_set = np.concatenate(lableset, axis=0)
        if(including_ring == False):
            self.point_set = self.point_set[:, :, 0:num_channel]
        
        else:
            temp = list(range(num_channel))
            temp.append(4)
            self.point_set = self.point_set[:, :, temp]
        
        
        self.num_channel = num_channel
        print("num_channel:", num_channel)
        print("point_set:", self.point_set.shape)
        print("lable_set:", self.label_set.shape)

        labelweights, _ = np.histogram(self.label_set, range(3))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("self.labelweights", self.labelweights)

    def __getitem__(self, index):

        # if(self.is_test == False):
        point_set = self.point_set[index]
        label_set = self.label_set[index]

        point_set[:, :3] = point_set[:, :3] - np.expand_dims(np.mean(point_set[:, :3], axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)

        # if(dist == 0):
        # print(point_set)
        # print(dist)
        # dist = LA.norm(point_set, axis=1)

        point_set[:, :3] = point_set[:, :3] / dist  # scale

        if(self.is_train == True):
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)  # random rotation

            multiplier = np.random.uniform(1, 2)
            point_set[:, 0] = point_set[:, 0] * multiplier  # random scaling

            symer = np.random.choice([-1, 1])
            point_set[:, 2] = point_set[:, 2] * symer  # random sysmtrilizing

            point_set[:, :3] += np.random.normal(0, 0.02, size=point_set[:, :3].shape)  # random jitter

        # duplicated
        #         pos_num = np.sum(label_set == 1)
        #         neg_num = np.sum(label_set == 0)

        #         multi = np.random.uniform(2, 5)

        #         pos_select_pos = (int)(20000/multi)
        #         pos_select_neg = 20000 - pos_select_pos

        #         if(neg_num > pos_select_neg):
        #             select_neg = np.random.choice(neg_num, pos_select_neg, replace=False)
        #             select_pos = np.random.choice(pos_num, pos_select_pos, replace=True)

        #             point_pos = point_set[label_set == 1, :]
        #             point_pos_new = point_pos[select_pos, :]

        #             label_set_pos = label_set[label_set == 1]
        #             label_pos_new = label_set_pos[select_pos]

        #             point_neg = point_set[label_set == 0, :]
        #             point_neg_new = point_neg[select_neg, :]

        #             label_set_neg = label_set[label_set == 0]
        #             label_neg_new = label_set_neg[select_neg]

        #             point_set = np.vstack((point_pos_new, point_neg_new))
        #             label_set = np.hstack((label_pos_new, label_neg_new))

        point_set = torch.from_numpy(point_set).float()
        seg = torch.from_numpy(label_set).long()

        return point_set, seg

        # else:
        #     point_set = self.point_set[index]
        #     label_set = self.label_set[index]
        #     point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        #     dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        #     point_set = point_set / dist  # scale
        #     point_set = torch.from_numpy(point_set).float()
        #     seg = torch.from_numpy(label_set).long()
        #     return point_set,seg


    def __len__(self):
        return len(self.point_set)
