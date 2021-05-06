# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:14:30 2021

@author: nadin
"""
import os
import glob
from matplotlib import pyplot as plt
from skimage.io import imread
import xlsxwriter
import numpy as np
from Net3 import Net3
import torch

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

    


def SPARCK():
     # Obtain folder
    folder = os.getcwd()
    
    # Load all images in the folder 'images'
    img_paths = glob.glob(os.path.join(folder+'\\images', 'img_*.png'))
    img_paths = sorted(img_paths) 
    
    
    # Store the images with corner predictions in the folder 'results'
    a =  str(folder+'\\predictions_SPARCK\\')
    save_path = folder+'\\SPARCK\\models'
    
    if not os.path.exists(a):
        os.makedirs(a)
    
    print('Loaded', len(img_paths), 'images.')   
    
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    
    workbook = xlsxwriter.Workbook(folder+'\\predictions_SPARCK\\gate_points.xlsx')
    worksheet = workbook.add_worksheet("gate_points")
    
    row = 0
    
    in_channels = 3
    hidden_channels3 = [5, 5, 5]
    out_features = 2
    
    n_samples = 2500
    feat_size = 16
    
    
    net3 = Net3(in_channels, hidden_channels3, out_features)
    net3.load_state_dict(torch.load(os.path.join(save_path,'net3.pth')))
    net3.eval()
    
    for i in range(0, len(img_paths)):
        # Obtain image
        img_path = img_paths[i]
        img_name = image_name(img_path)
        X = imread(img_path) 
        
        gate_points = eval_img(i, img_paths, n_samples, feat_size, net3)
        
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

SPARCK()