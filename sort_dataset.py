import os
import os.path as osp
import shutil
import numpy as np
import h5py
import torch
import pdb
from pathlib import Path
import shutil
from glob import glob

def xyz2rtp(xyz):
    #points [x,y,z,laserID]
    #NxMx4
    #convert points(xyz arrary) to polar coordinates
    eps = 1e-10
    rtp = np.zeros((xyz.shape[0],xyz.shape[1],3))
    #r
    rtp[:,:,0] = np.linalg.norm(xyz[:,:,:3],axis=2)
    #theta
    rtp[:,:,1] = np.arctan(xyz[:,:,1]/(xyz[:,:,0]+eps))
    #phi is ralated to laserID
    #phi
    rtp[:,:,2] = np.round(np.arccos(xyz[:,:,2]/(rtp[:,:,0]+eps)),6)
    return rtp

def return_sort_dict(laserID):
    new_dict = []
    laserID = torch.tensor(laserID)
    for i in laserID:
        this_ring = i
        convert_idx= this_ring.sort()[1]
        new_dict.append((convert_idx).numpy())
    return np.array(new_dict)
def sort_data(data,my_dict):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i][my_dict[i]])
    return np.array(new_data)
    


root = '../bigRed_h5_pointnet'
experiment_dir = Path('../bigRed_h5_pointnet_sorted')
experiment_dir.mkdir(exist_ok=True)

txt_list = glob('../bigRed_h5_pointnet/*.txt')
for item in txt_list:
    shutil.copy(item, str(experiment_dir))

with open(os.path.join(root, "all_files.txt"), 'r') as f:
    data_list = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

print('Add rtp...')
for file_name in data_list:
    file_root = os.path.join(root,file_name)
    print(file_root)
    f = h5py.File(file_root, 'r+') 
    intensity = np.array(f['intensity'])
    label = np.array(f['label'])
    laserID = np.array(f['laserID'])
    xyz = np.array(f['xyz'])
    rtp = xyz2rtp(xyz)
    f.create_dataset("rtp", data=rtp)
    f.close()
    
print('sort...')
for file_name in data_list:
    file_root = os.path.join(root,file_name)
    print(file_root)
    f = h5py.File(file_root, 'r+') 
    intensity = np.array(f['intensity'])
    label = np.array(f['label'])
    laserID = np.array(f['laserID'])
    xyz = np.array(f['xyz'])
    rtp = np.array(f['rtp'])
    f.close()
    new_dict = return_sort_dict(laserID)
    new_intensity = sort_data(intensity,new_dict)
    new_label = sort_data(label,new_dict)
    new_laserID = sort_data(laserID,new_dict)
    new_xyz = sort_data(xyz,new_dict)
    new_rtp = sort_data(rtp,new_dict)
    f = h5py.File(os.path.join(experiment_dir,file_name), 'w')
    f.create_dataset('intensity',data=new_intensity)
    f.create_dataset('label',data=new_label)
    f.create_dataset('laserID',data=new_laserID)
    f.create_dataset('xyz',data=new_xyz)
    f.create_dataset('rtp',data=new_rtp)
    f.close()

    