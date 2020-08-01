cd ../
apt-get update
apt-get install wget
apt-get install zip
apt-get install curl
wget https://www.dropbox.com/s/hbnpnchf8mrzfz1/bigRed_h5_pointnet_sorted.tar.gz
tar -xvzf bigRed_h5_pointnet_sorted.tar.gz
rm -r bigRed_h5_pointnet_sorted.tar.gz
git clone https://github.com/andreafabrizi/Dropbox-Uploader.git
cd ~/Dropbox-Uploader
sudo chmod +x dropbox_uploader.sh
cd 


pip install tqdm
pip install pandas
pip install wandb
pip install h5py
pip install kornia
apt-get install watch
pip install gpustat


# install anaconda3.
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
#bash Anaconda3-2019.07-Linux-x86_64.sh


#source ~/.bashrc


# make sure system cuda version is the same with pytorch cuda
# follow the instruction of Pyotrch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# export PATH=/usr/local/cuda-10.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

# conda create -n deepgcn
# conda activate deepgcn
# # make sure pytorch version >=1.4.0
# conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 tensorflow=1.14.0 python=3.7 -c pytorch

# # command to install pytorch geometric
# pip install --verbose --no-cache-dir torch-scatter
# pip install --verbose --no-cache-dir torch-sparse
# pip install --verbose --no-cache-dir torch-cluster
# pip install --verbose --no-cache-dir torch-spline-conv
# pip install torch-geometric
# pip install --upgrade tensorflow-graphics
# # install useful modules
# pip install requests # sometimes pytorch geometric forget to install it, but it is in need
# pip install tqdm
