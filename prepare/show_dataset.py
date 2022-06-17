import os
from glob import glob
from random import random
import numpy as np
import cv2 as cv

def img_with_gt_boxes(labels, img_path):
    img = cv.imread(img_path)

    with open(labels, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.replace("\n","").split(" ")
        object = line[0]
        box = [int(element) for element in line[1:]]
        p1 = [box[0]-box[2]//2, box[1]-box[3]//2]
        p2 =[box[0]+box[2]//2, box[1]+box[3]//2]
        img = cv.rectangle(img, p1, p2, color=(0,0,255), thickness=2)
        img = cv.putText(img, object, (p1[0],p1[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        
    return img

def main():
    DATASET_PATH = "../dataset"
    img_paths = glob(os.path.join(DATASET_PATH, "images/*"))
    label_paths = [path.replace(".jpg", ".txt").replace("images", "labels") for path in img_paths]
    data_size = len(img_paths)

    random_indexes = np.random.randint(0,data_size, 5)
    img_to_show = [img_paths[id] for id in random_indexes]
    labels_to_show = [label_paths[id] for id in random_indexes]

    list_imgs = []
    for img_path, label_path in zip(img_to_show, labels_to_show):
        img = cv.imread(img_path)
        img = cv.resize(img, (350, 350))
        img_with_boxes = img_with_gt_boxes(label_path, img_path)
        img_with_boxes = cv.resize(img_with_boxes, (350, 350))
        img = np.vstack((img, img_with_boxes))
        list_imgs.append(img)

    img = np.hstack(tuple(list_imgs))
    cv.imshow("example", img)
    cv.waitKey(0)

    return

if __name__ == "__main__":
    main()