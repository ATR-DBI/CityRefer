""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/model_util_scannet.py
"""

import numpy as np
import pandas as pd
import sys
import os
import torch

sys.path.append(os.path.join(os.getcwd(), os.pardir, "lib")) # HACK add the lib folder

from lib.config import CONF

sys.path.append(os.path.join(os.getcwd(), os.pardir, "utils")) # HACK add the lib folder
from box_util import get_3d_box

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

def rotate_aligned_boxes_along_axis(input_boxes, rot_mat, axis):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))

    if axis == "x":     
        d1, d2 = lengths[:,1]/2.0, lengths[:,2]/2.0
    elif axis == "y":
        d1, d2 = lengths[:,0]/2.0, lengths[:,2]/2.0
    else:
        d1, d2 = lengths[:,0]/2.0, lengths[:,1]/2.0

    new_1 = np.zeros((d1.shape[0], 4))
    new_2 = np.zeros((d1.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((d1.shape[0], 3))
        crnrs[:,0] = crnr[0]*d1
        crnrs[:,1] = crnr[1]*d2
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_1[:,i] = crnrs[:,0]
        new_2[:,i] = crnrs[:,1]
    
    new_d1 = 2.0*np.max(new_1, 1)
    new_d2 = 2.0*np.max(new_2, 1)    

    if axis == "x":     
        new_lengths = np.stack((lengths[:,0], new_d1, new_d2), axis=1)
    elif axis == "y":
        new_lengths = np.stack((new_d1, lengths[:,1], new_d2), axis=1)
    else:
        new_lengths = np.stack((new_d1, new_d2, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

class SensatUrbanDatasetConfig(object):
    def __init__(self):
        # self.type2class = {
        #     0: 'Ground', 1: 'High Vegetation', 2: 'Buildings', 3: 'Walls',
        #     4: 'Bridge', 5: 'Parking', 6: 'Rail', 7: 'traffic Roads', 8: 'Street Furniture',
        #     9: 'Cars', 10: 'Footpath', 11: 'Bikes', 12: 'Water'
        # }
        
        # label_id -> label_name
        #labelmap = {0: 'Ground', 2: 'Buildings', 5: 'Parking', 9: 'Cars'} 
        labelmap = {0: 'Ground', 2: 'Building', 5: 'Parking', 9: 'Car'}  # see data/sensaturban/meta_data/sensaturban-labels.tsv
        # type2class: class_ind -> label_name
        self.type2class = {}
        self.label_ids = []
        # label_id2class: label_id -> class_ind
        self.label_id2class = {}
        for class_ind, label_id in enumerate(sorted(labelmap.keys())):
            label_name = labelmap[label_id]
            self.type2class[class_ind] = label_name
            self.label_ids.append(label_id)
            self.label_id2class[label_id] = class_ind
        # class_ind -> label_id
        self.label_ids = np.array(self.label_ids)
        #self.type2class['others'] = len(self.type2class) 
        
        self.num_class = len(self.type2class.keys())
        self.num_heading_bin = 1
        self.num_size_cluster = len(self.type2class.keys())        
        
        bbox_size_df = pd.read_pickle(os.path.join(CONF.PATH.SCAN_META, 'bbox_size.pkl'))
        bbox_size_df['class_ind'] = bbox_size_df.label_id.map(self.label_id2class)
        # 縦, class_ind, 横 x_len, y_len, z_len
        self.mean_size_arr = np.array(bbox_size_df.groupby('class_ind').mean().reset_index().sort_values(['class_ind'])[['x_len', 'y_len', 'z_len']])
        
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.
        
        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return np.zeros(pred_cls.shape[0])

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_batch(self, pred_cls, residual):
        ''' Inverse function to size2class '''      
        return self.mean_size_arr[pred_cls] + residual

    def class2size_torch(self, pred_cls, residual):
        mean_size_arr = torch.Tensor(self.mean_size_arr).cuda()
        ''' Inverse function to size2class '''
        return mean_size_arr[pred_cls] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2obb_batch(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle_batch(heading_class, heading_residual)
        box_size = self.class2size_batch(size_class, size_residual)
        obb = np.zeros((heading_class.shape[0], 7))
        obb[:, 0:3] = center
        obb[:, 3:6] = box_size
        obb[:, 6] = heading_angle*-1
        return obb
