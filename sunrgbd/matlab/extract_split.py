from scipy import io
import cv2 as cv
import numpy as np

hash_train = {}
hash_val = {}

split = io.loadmat("/home/luc/Documents/votenet/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat")

N_train = split["alltrain"].shape[1]
N_val = split["alltest"].shape[1]

for i in range(N_train):
    folder_path = split["alltrain"][0][i].item()
    folder_path = "../OFFICIAL_SUNRGBD" + folder_path[16:]
    hash_train[folder_path] = i

for i in range(N_val):
    folder_path = split["alltest"][0][i].item()
    folder_path = "../OFFICIAL_SUNRGBD" + folder_path[16:]
    hash_val[folder_path] = i

SUNRGBDMeta = io.loadmat("/home/luc/Documents/votenet/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat")["SUNRGBDMeta"][0]

fid_train = open('../sunrgbd_trainval/train_data_idx.txt', 'w')
fid_val = open('../sunrgbd_trainval/val_data_idx.txt', 'w')

for imageId in range(10335):
    data = SUNRGBDMeta[imageId]
    depthpath = data[3].item()
    depthpath = "../OFFICIAL_SUNRGBD" + depthpath[16:]
    filepath = "/".join(depthpath.split("/")[:-2])
    if filepath in list(hash_train.keys()):
        fid_train.write(f"{imageId}\n")
    elif filepath in list(hash_val.keys()):
        fid_val.write(f"{imageId}\n")
    else:
        a = 1

fid_train.close()
fid_val.close()
