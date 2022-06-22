import os
from glob import glob
import json
import shutil
from tqdm import tqdm
import cv2
import numpy as np

class SUNRGBD_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3,3), order='F').T
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3,3), order='F').T
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,0:3])) # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)
        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]
        return uv[:,0:2], pc2[:,2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir)

        self.num_samples = 10335

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, f"{idx}.jpg")
        return load_image(img_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, f"{idx}.txt")
        return SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, f"{idx}.txt")
        return read_sunrgbd_label(label_filename)

class SUNObject3d(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.centroid = np.array([data[1],data[2],data[3]])
        self.w = data[4]
        self.l = data[5]
        self.h = data[6]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[7]
        self.orientation[1] = data[8]
        self.heading_angle = -1 * np.arctan2(self.orientation[1], self.orientation[0])

def load_image(img_filename):
    return cv2.imread(img_filename)

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

def draw_projected_box3d(image, qs, label, color=(255,0,0), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
        #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i,j=k,(k+1)%4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

        i,j=k+4,(k+1)%4 + 4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        i,j=k,k+4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
        cv2.putText(image, label, (qs[5,0], qs[5,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return image

def read_sunrgbd_label(label_filename):
    with open(label_filename, "r") as f:
        objects = f.read().split("\n")[:-1]
        objects = [SUNObject3d(obj) for obj in objects]
    return objects

def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotz(-1*obj.heading_angle)
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj.l # along heading arrow
    w = obj.w # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d,_ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_orientation_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in image coord.
            orientation_3d: (2,3) array in depth coord.
    '''
    
    # orientation in object coordinate system
    ori = obj.orientation
    orientation_3d = np.array([[0, ori[0]],[0, ori[1]],[0,0]])
    center = obj.centroid
    orientation_3d[0,:] = orientation_3d[0,:] + center[0]
    orientation_3d[1,:] = orientation_3d[1,:] + center[1]
    orientation_3d[2,:] = orientation_3d[2,:] + center[2]
    
    # project orientation into the image plane
    orientation_2d,_ = calib.project_upright_depth_to_image(np.transpose(orientation_3d))
    return orientation_2d, np.transpose(orientation_3d)

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def data_viz(data_dir, dump_dir):  
    ''' Examine and visualize SUN RGB-D data. '''
    sunrgbd = sunrgbd_object(data_dir)
    idxs = np.array(range(1,len(sunrgbd)+1))
    np.random.seed(1)
    np.random.shuffle(idxs)
    for idx in range(len(sunrgbd)):
        data_idx = idxs[idx]
        print('-'*10, 'data index: ', data_idx)

        # get calibration parameters
        calib = sunrgbd.get_calibration(data_idx)

        # Load box labels
        objects = sunrgbd.get_label_objects(data_idx)
        print('Objects:', objects)

        # Dump OBJ files for 3D bounding boxes
        # l,w,h correspond to dx,dy,dz
        # heading angle is from +X rotating towards -Y
        # (+X is degree, -Y is 90 degrees)
        oriented_boxes = []
        for obj in objects:
            obb = np.zeros((7))
            obb[0:3] = obj.centroid
            # Some conversion to map with default setting of w,l,h
            # and angle in box dumping
            obb[3:6] = np.array([obj.l,obj.w,obj.h])*2
            obb[6] = -1 * obj.heading_angle
            print('Object cls, heading, l, w, h:',\
                 obj.classname, obj.heading_angle, obj.l, obj.w, obj.h)
            oriented_boxes.append(obb)
        if len(oriented_boxes)>0:
            oriented_boxes = np.vstack(tuple(oriented_boxes))
            # write_oriented_bbox(oriented_boxes,
            #     os.path.join(dump_dir, 'obbs.ply'))
        else:
            print('-'*30)
            continue

        # Draw 3D boxes projections on the image
        box3d = []
        ori3d = []
        box3d_image = []
        labels = []
        for obj in objects:
            label = obj.classname
            labels.append(label)
            corners_3d_image, corners_3d = compute_box_3d(obj, calib)
            ori_3d_image, ori_3d = compute_orientation_3d(obj, calib)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
            ori3d.append(ori_3d)
            box3d_image.append(corners_3d_image)

        # show bb3d in image
        img = sunrgbd.get_image(data_idx)

        for box, label in zip(box3d_image, labels):
            img = draw_projected_box3d(img, box, label)
        cv2.imwrite(f"/home/luc/Documents/sunrgbd_dataset/prepare/test_image/{data_idx}.jpg", img)
        
        pc_box3d = np.concatenate(box3d, 0)
        pc_ori3d = np.concatenate(ori3d, 0)
        print(pc_box3d.shape)
        print(pc_ori3d.shape)
        # pc_util.write_ply(pc_box3d, os.path.join(dump_dir, 'box3d_corners.ply'))
        # pc_util.write_ply(pc_ori3d, os.path.join(dump_dir, 'box3d_ori.ply'))
        print('-'*30)
        print('Point clouds and bounding boxes saved to PLY files under %s'%(dump_dir))
        print('Type anything to continue to the next sample...')
        input()

def main():

    BASE_DIR = "../sunrgbd/sunrgbd_trainval"
    dump_dir=os.path.join(BASE_DIR, "data_viz.dump")
    OUTPUT_FOLDER = "../dataset"

    img_path = "../sunrgbd/sunrgbd_trainval/image"
    label_path = "../sunrgbd/sunrgbd_trainval/label"  

    data_viz(BASE_DIR, dump_dir)

    return


if __name__ == "__main__":
    main()