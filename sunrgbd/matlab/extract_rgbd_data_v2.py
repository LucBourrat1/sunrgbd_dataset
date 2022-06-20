from re import X
from scipy import io
import numpy as np
import cv2 as cv
import os

def read_3d_pts_general(depthVis, K, imsize, rgbpath):
    cx = K[0,2]; cy = K[1,2]
    fx = K[0,0]; fy = K[1,1]
    im= cv.imread(rgbpath)
    x, y =np.meshgrid(range(1,imsize[1]+1), range(1,imsize[0]+1))
    x3 = (x-cx) * depthVis[:,:,0] * 1/fx
    y3 = (y-cy) * depthVis[:,:,0] * 1/fy
    z3 = depthVis[:,:,0]

    
    
    print("debug")
    return

def read3dPoints(data, depthpath, rgbpath):
    depthVis = cv.imread(depthpath)
    imsize = depthVis.shape
    K =data[2]
    [rgb, points3d] = read_3d_pts_general(depthVis, K, imsize, rgbpath)

    return

def main():
    SUNRGBDMeta = io.loadmat("/home/luc/Documents/votenet/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat")["SUNRGBDMeta"][0]
    SUNRGBDMeta2DBB = io.loadmat("/home/luc/Documents/votenet/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat")["SUNRGBDMeta2DBB"][0]

    # create folders
    depth_folder = "../sunrgbd_trainval/depth/"
    image_folder = "../sunrgbd_trainval/image/"
    calib_folder = "../sunrgbd_trainval/calib/"
    det_label_folder = "../sunrgbd_trainval/label/"
    seg_label_folder = "../sunrgbd_trainval/seg_label/"
    if not os.path.exists(depth_folder):
        os.mkdir(depth_folder)
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    if not os.path.exists(calib_folder):
        os.mkdir(calib_folder)
    if not os.path.exists(det_label_folder):
        os.mkdir(det_label_folder)
    if not os.path.exists(seg_label_folder):
        os.mkdir(seg_label_folder)

    # read
    for imageId in range(10335):
        depthpath = SUNRGBDMeta[imageId][3].item()
        #depthpath1 = "../OFFICIAL_SUNRGBD" + depthpath[16:]
        depthpath = "../OFFICIAL_SUNRGBD" + depthpath[16:].replace("depth", "depth_bfx")
        rgbpath = SUNRGBDMeta[imageId][4].item()
        rgbpath = "../OFFICIAL_SUNRGBD" + rgbpath[16:]

        [rgb, points3d, depthInpaint, imsize] = read3dPoints(SUNRGBDMeta[imageId], depthpath, rgbpath)

    return

if __name__ == "__main__":
    main()