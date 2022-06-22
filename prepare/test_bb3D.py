import cv2 as cv
import json
import numpy as np


img_path = "/home/luc/Documents/sunrgbd_dataset/prepare/test_image_103/0000103.jpg"
#label_path = "/home/luc/Documents/votenet/luc/annotation3D.json"


img = cv.imread(img_path)

extrinsic = np.array(
    [[0.979589, 0.200614, 0.012593, 0.000000],
    [-0.200614, 0.971820, 0.123772, 0.000000],
    [0.012593, -0.123772, 0.992231, 0.000000]]
    )

intrinsic = np.array([
    [529.500000, 0.000000, 365.000000],
    [0.000000, 529.500000, 265.000000],
    [0.000000, 0.000000, 1.000000]
    ])

p1 = np.array([[1.0473071503499998, 4.168695787, -0.24685933444999997, 1]]).T
p2 = np.array([[1.0473071503499998, -0.24685933444999997, -4.168695787, 1]]).T
# p3 = np.array([[0.5599692605,  2.426446281, -0.6590909091, 1]]).T
# p4 = np.array([[-0.2868617169, 2.092170895, -0.6590909091, 1]]).T
# p5 = np.array([[-0.15, 1.745454545, 1.077272727, 1]]).T
# p6 = np.array([[0.6968309774, 2.079729931, 1.077272727, 1]]).T
# p7 = np.array([[0.5599692605,  2.426446281, 1.077272727, 1]]).T
# p8 = np.array([[-0.2868617169, 2.092170895, 1.077272727, 1]]).T

#points = [p1, p2, p3, p4, p5, p6, p7, p8]
points = [p1, p2]

princ_point = np.array([[img.shape[0], img.shape[1], 0]]).T

points_2d = [intrinsic@extrinsic@p for p in points]

print(f"p1:\n{points_2d[0]}")
print(f"p2:\n{points_2d[1]}")
# print(f"p3:\n{points_2d[2]}")
# print(f"p4:\n{points_2d[3]}")
# print(f"p5:\n{points_2d[4]}")
# print(f"p6:\n{points_2d[5]}")
# print(f"p7:\n{points_2d[6]}")
# print(f"p8:\n{points_2d[7]}")

for point in points_2d:
    img = cv.circle(img, (int(point[0][0]), int(point[1][0])), 5, (255,0,0), 3)

img = cv.circle(img, (20,20), 5, (255,0,0), 3)

cv.imshow("image", img)
cv.waitKey(0)

print(img.shape)

print("debug")




