# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:14:30 2021

@author: nadin
"""
from predict import get 

import os
import glob
from matplotlib import pyplot as plt
from skimage.io import imread
import xlsxwriter
import numpy as np
  

def image_name(img_path):
    
        # Return image name from path
        head, tail = os.path.split(img_path)
        return tail
    


def MeanBlue():
     # Obtain folder
    folder = os.getcwd()
    
    # Load all images in the folder 'images'
    img_paths = glob.glob(os.path.join(folder+'\\images', 'img_*.png'))
    img_paths = sorted(img_paths) 
    
    
    # Store the images with corner predictions in the folder 'results'
    a =  str(folder+'\\predictions_MeanBlue\\')
    if not os.path.exists(a):
        os.makedirs(a)
    
    print('Loaded', len(img_paths), 'images.')   
    
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
    
    workbook = xlsxwriter.Workbook(folder+'\\predictions_MeanBlue\\corners.xlsx')
    worksheet = workbook.add_worksheet("corners")
    
    row = 0
    
    for i in range(0, len(img_paths)):
        # Obtain image
        img_path = img_paths[i]
        img_name = image_name(img_path)
        X = imread(img_path) 
        
        corners_pred, centers, img_test, obstacle_detected = get(X)

        # Plot corners
        if obstacle_detected:
            plt.scatter(corners_pred[:,0], corners_pred[:,1], c = 'r', s = 500)
        plt.imshow(X)
        plt.axis('off')

        # Save image
        b =  str(a+img_name)
        plt.savefig(b, bbox_inches='tight',pad_inches = 0)
        plt.close()
        
        write_excel = np.hstack([img_name, corners_pred.flatten()])
        for col, value in enumerate(write_excel):
            worksheet.write(row, col, value)
    
        row += 1

        # Print progress
        if i % 15 == 0 and i != 0:
            print(' ', int(i / len(img_paths) * 100), '%')
        elif i == 0:
            print('Start Processing')
        else:
            print("=",  end = '')
            
        

    workbook.close()
    print('     100%')

MeanBlue()