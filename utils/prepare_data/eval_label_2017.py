import os
import numpy as np
import math
import cv2 as cv

'''
本函数将 gt为8个值的标签，转化为 4个值的标签(xmin, ymin, xmax, ymax)
'''

gt_path = '/home/north-computer/Text/data/2017/val_gt'
out_path = 'gt_4'
if not os.path.exists(out_path):
    os.makedirs(out_path)

files = os.listdir(gt_path)
files.sort()
#files=files[:100]
for gt_file in files:
    _, basename = os.path.split(gt_file)
    stem, ext = os.path.splitext(basename)
    gt_file = os.path.join(gt_path, gt_file)

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        splitted_line = line.strip().lower().split(',')
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        pt_x[0, 0] = int(splitted_line[0])
        pt_y[0, 0] = int(splitted_line[1])
        pt_x[1, 0] = int(splitted_line[2])
        pt_y[1, 0] = int(splitted_line[3])
        pt_x[2, 0] = int(splitted_line[4])
        pt_y[2, 0] = int(splitted_line[5])
        pt_x[3, 0] = int(splitted_line[6])
        pt_y[3, 0] = int(splitted_line[7])

        ind_x = np.argsort(pt_x, axis=0)
        ind_y = np.argsort(pt_y, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_y]

        xmin = int(pt_x[0, 0])
        ymin = int(pt_y[0, 0])
        xmax = int(pt_x[3, 0])
        ymax = int(pt_y[3, 0])

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, stem) + '.txt', 'a') as f:
            line = ','.join([str(xmin),str(ymin),str(xmax),str(ymax)])+',###\r\n'
            f.write(line)

