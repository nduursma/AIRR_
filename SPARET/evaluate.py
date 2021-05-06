# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:40:43 2021

@author: nadin
"""

from matplotlib import pyplot as plt
import numpy as np
import skimage.transform
import random

from sklearn.cluster import DBSCAN
random.seed(20)

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
    
    patch = image[left:right, top:bottom, :]
    return patch.flatten()






def eval_img(idx, img_paths, n_samples, feat_size, clf):

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
    
    # Obtain labels from CNN
    y_data = clf.predict(X_data)
    gate_points = points[y_data == 1]*2
    
    # Filter outliers
    db = DBSCAN(eps=16, min_samples=6).fit(gate_points)
    labels = db.labels_ 
    gate_points_no_noise = gate_points[labels != -1]
    
    return gate_points_no_noise

