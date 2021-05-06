# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:14:30 2021

@author: nadin
"""
from matplotlib import pyplot as plt
import numpy as np

import os
import glob
import random

import pickle

random.seed(20)

import xlsxwriter

from evaluate import eval_img

def image_name(img_path):
    
        # Return image name from path
        head, tail = os.path.split(img_path)
        return tail
# Show sampled points
def show_points(points, img):
    
    plt.figure()
    plt.imshow(img)
    plt.scatter(points[:,0], points[:,1], s = 40, c = 'r')
    plt.axis('off')

    return plt

    


def SPARET():
     # Obtain folder
    folder = os.getcwd()
    
    # Load all images in the folder 'images'
    img_paths = glob.glob(os.path.join(folder+'\\images', 'img_*.png'))
    img_paths = sorted(img_paths) 
    
    
    # Store the images with corner predictions in the folder 'results'
    a =  str(folder+'\\predictions_SPARET\\')
    save_path = folder+'\\SPARET\\models'
    
    if not os.path.exists(a):
        os.makedirs(a)
    
    print('Loaded', len(img_paths), 'images.')   
    
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    
    workbook = xlsxwriter.Workbook(folder+'\\predictions_SPARET\\gate_points.xlsx')
    worksheet = workbook.add_worksheet("gate_points")
    
    row = 0
    
    
    n_samples = 2500
    feat_size = 16
    
    with open(save_path+'\\et_clf.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    for i in range(0, len(img_paths)):
        # Obtain image
        img_path = img_paths[i]
        img_name = image_name(img_path)
        X = plt.imread(img_path) 
        
        gate_points = eval_img(i, img_paths, n_samples, feat_size, clf)
        
        plt.imshow(X)
        plt.scatter(gate_points[:,0], gate_points[:,1], c = 'r', s = 40)
        plt.axis('off')  

        # Save image
        b =  str(a+img_name)
        plt.savefig(b, bbox_inches='tight',pad_inches = 0)
        plt.close()
        
        write_excel_x = np.hstack([img_name+'_x', gate_points[:,0].flatten()])
        write_excel_y = np.hstack([img_name+ '_y', gate_points[:,1].flatten()])
        for col, value in enumerate(write_excel_x):
            worksheet.write(row, col, value)
        for col, value in enumerate(write_excel_y):
            worksheet.write(row+1, col, value)
            
    
        row += 2

        # Print progress
        if i % 15 == 0 and i != 0:
            print(' ', int(i / len(img_paths) * 100), '%')
        elif i == 0:
            print('Start Processing')
        else:
            print("=",  end = '')
            
        

    workbook.close()
    print('     100%')

SPARET()