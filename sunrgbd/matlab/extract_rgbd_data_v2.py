from re import X
from scipy import io
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import shutil
from glob import glob

def read_3d_pts_general(depthVis, K, imsize, rgbpath):
    cx = K[0,2]; cy = K[1,2]
    fx = K[0,0]; fy = K[1,1]
    im = cv.imread(rgbpath)
    x, y = np.meshgrid(range(1,imsize[1]+1), range(1,imsize[0]+1))
    x3 = (x-cx) * depthVis[:,:,0] * 1/fx
    y3 = (y-cy) * depthVis[:,:,0] * 1/fy
    z3 = depthVis[:,:,0]
    points3d = np.array([x3, z3, -y3])

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
    for imageId in tqdm(range(10335)):

        # read depth and rgb and make point cloud
        depthpath = SUNRGBDMeta[imageId][3].item()
        #depthpath1 = "../OFFICIAL_SUNRGBD" + depthpath[16:]
        depthpath = "../OFFICIAL_SUNRGBD" + depthpath[16:].replace("depth", "depth_bfx")
        rgbpath = SUNRGBDMeta[imageId][4].item()
        rgbpath = "../OFFICIAL_SUNRGBD" + rgbpath[16:]
        shutil.copy(rgbpath, os.path.join(image_folder, f"{imageId}.jpg"))
        shutil.copy(depthpath, os.path.join(depth_folder, f"{imageId}.png"))

        #[rgb, points3d, depthInpaint, imsize] = read3dPoints(SUNRGBDMeta[imageId], depthpath, rgbpath)

        # write calibration files
        # Rtilt = SUNRGBDMeta[imageId][5]
        extrinsic_path = glob("/".join(rgbpath.split("/")[:-2]) + "/extrinsics/*")[0]
        with open(extrinsic_path, "r") as f:
            l1 = f.readline().split("\n")[:-1]
            l2 = f.readline().split("\n")[:-1]
            l3 = f.readline().split("\n")[:-1]
            l1 = np.array([float(x) for x in l1[0].split(" ")[:-1]])
            l2 = np.array([float(x) for x in l2[0].split(" ")[:-1]])
            l3 = np.array([float(x) for x in l3[0].split(" ")[:-1]])
            Rtilt = np.vstack((l1, l2, l3))
            
        K = SUNRGBDMeta[imageId][2]
        mat = np.vstack((Rtilt, K)).reshape(2,9)
        np.savetxt(os.path.join(calib_folder, f"{imageId}.txt"), mat, fmt="%f" , delimiter=" ")

        # write 3D bounding boxes annotations
        with open(os.path.join(det_label_folder, f"{imageId}.txt"), "w") as f:
            for j in range(SUNRGBDMeta[imageId][10].shape[1]):
                centroid = SUNRGBDMeta[imageId][10][0][j][2][0]
                classname = SUNRGBDMeta[imageId][10][0][j][3][0]
                orientation = SUNRGBDMeta[imageId][10][0][j][5][0]
                coeffs = SUNRGBDMeta[imageId][10][0][j][1][0]
                # box2d = SUNRGBDMeta2DBB[1][1][0][j][1][0]
                # bb2d_name = SUNRGBDMeta2DBB[1][1][0][j][2][0]
                f.write(f"{classname} {centroid[0]} {centroid[1]} {centroid[2]} {coeffs[0]} {coeffs[1]} {coeffs[2]} {orientation[0]} {orientation[1]}\n")

    return

if __name__ == "__main__":
    main()