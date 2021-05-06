# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:40:43 2021

@author: nadin
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from skimage import color
from skimage import io

import os
import glob
import skimage.transform
import random

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_curve, auc

random.seed(20)
torch.cuda.is_available()

feat_size = 16

def get_patches_eval(points_arr, img):
    
    # Get patches for one image
    patch_lst = []
    
    for i, point in enumerate(points_arr):
        patch_lst.append(get_patch_at_point_eval(img, point))
        
    return np.array(patch_lst)

def get_patch_at_point_eval(image, point): 
    
    # Get patch at point
    top = int(point[0] - feat_size/2)
    bottom = int(point[0] + feat_size/2)
    left = int(point[1] - feat_size/2)
    right = int(point[1] + feat_size/2)
    
    return image[left:right, top:bottom, :]



def eval_img(idx, img_paths, n_samples, feat_size, net3):

    # Find all points on gate
    img = plt.imread(img_paths[idx]) 
    img_resize = skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=False)
    
    # Reshape
    img_size_x = img_resize.shape[1]
    img_size_y = img_resize.shape[0]

    # Create grid points
    Xgrid = np.linspace(feat_size/2, img_size_x - feat_size/2, int(np.sqrt(n_samples)))
    Ygrid = np.linspace(feat_size/2, img_size_y - feat_size/2, int(np.sqrt(n_samples)))

    grid = np.array(np.meshgrid(Xgrid, Ygrid))
    points = grid.reshape(2,-1).T

    # Put in data vector for evaluation
    X_data = get_patches_eval(points, img_resize)
    X_data = X_data.reshape(X_data.shape[0], 3, feat_size, feat_size)
    X_data  = torch.from_numpy(X_data)
    
    # Obtain labels from CNN
    y_data = net3(X_data)
    y_data = y_data.argmax(-1)
    gate_points = points[y_data == 1]*2
    
    # Filter outliers
    db = DBSCAN(eps=16, min_samples=6).fit(gate_points)
    labels = db.labels_ 
    gate_points_no_noise = gate_points[labels != -1]
    
    return gate_points_no_noise

