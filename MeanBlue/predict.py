# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:13:35 2021

@author: nadin
"""
import numpy as np
import cv2

from skimage import measure
from sklearn.neighbors import NearestCentroid

from skimage.color import label2rgb
import heapq


def get(X):
    hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([100,150,115])
    upper_blue = np.array([117,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel_d0 = np.vstack([np.ones((3,3), np.uint8),
                           np.zeros((3,3), np.uint8)])
    
    kernel_e1 = np.vstack([np.ones((6, 25), np.uint8),
                          np.zeros((5, 25), np.uint8)]).T
    
    kernel_c1 =  np.vstack([-np.ones((1,51), np.uint8),
                            np.zeros((1,51), np.uint8),
                            np.ones((1,51), np.uint8)]).T
    kernel_c1_norm = np.prod(kernel_c1.shape)
    kernel_c1 = kernel_c1/kernel_c1_norm

    
    kernel_d1 = np.vstack([np.ones((6, 21), np.uint8),
                           np.zeros((5, 21), np.uint8)])
    
    # Fill white spaces in blue part of obstacles
    dilation0 = cv2.dilate(mask,kernel_d0,iterations = 1)
    
    # Remove noise / small blue pixels
    median1 = cv2.medianBlur(dilation0, 11)
    
    # Remove thin horizontal blue parts
    erosion1 = cv2.erode(median1,kernel_e1, iterations = 1)
    
    # Convolute to detect vertical blue areas
    convolution1 = cv2.filter2D(erosion1,-1,kernel_c1)
    convolution1[convolution1 >= 100] = 255
    convolution1[convolution1 < 100] = 0
    
    # Make the thin areas detected thicker to give them a higher 'weight'
    dilation1 = cv2.dilate(convolution1,kernel_d1,iterations = 1)
    img_test = dilation1
    
    # Obtain labels per segment
    label_image = measure.label(img_test)
    
    # Convert to RGB for plots
    image_labels = label2rgb(label_image, image=img_test, bg_label=0)    
    
    # Obtain the number of segments
    n_obj = np.max(label_image)

    # If there are more than two segments, than an obstacle is detected.
    if n_obj < 2:
        obstacle_detected = False
    else: 
        obstacle_detected  = True
   
    # Obstain all obstacle pixels and their segment label
    if obstacle_detected:
        points_lst = []
        label_lst = []

        for y, row in enumerate(label_image):
            for x, label_i in enumerate(row):
                if label_i != 0:
                    points_lst.append([x,y])
                    label_lst.append(label_i)
        label_lst = np.array(label_lst)
        
        # Find the centroid and labels of each segment
        if points_lst != []:
            
            NC_clf = NearestCentroid()
            NC_clf.fit(points_lst, label_lst)
            centers = NC_clf.centroids_   
            
            n_points_lst = []
    
            # Determine the size of each segment
            for i in range(n_obj):
                
                n_points = label_lst[label_lst==i+1].shape[0]
                n_points_lst.append(n_points)

            # Find the two largest segment areas
            max_idxs = heapq.nlargest(2, np.arange(len(n_points_lst)), key=n_points_lst.__getitem__)
            center1 = centers[max_idxs[0]]
            center2 = centers[max_idxs[1]]
            
            # Determine the difference in y-coordinates of the centers of the two largest segments
            diff =  np.abs(center1[1] - center2[1])
            
            # If this is too big, than probably the wrong segment was selected
            if diff > 30:
                y_values = centers[:,1]
                diff_min = 10000 

            # Then we try to find two segments that are vertically aligned
                for i in range(-1, n_obj-1):
                    y1 = y_values[i]
                    y2 = y_values[i+1]
                    diff = np.abs(y1-y2)

                    if diff < diff_min:
                        
                        center1 = centers[i]
                        center2 = centers[i+1]
                        diff_min = diff

            # Order the centers, center1 = left, center2 = right
            if center1[0] > center2[0]:
                temp = center1
                center1 = center2
                center2 = temp

            centers = np.array([center1, center2])

            # Store the coordinates of the centers
            c1x = center1[0]
            c2x = center2[0]
            c1y = center1[1]
            c2y = center2[1]

            # Determine the vector from the left to the right center
            dx = (c2x - c1x)/2
            dy = (c2y -c1y)/2

            # Compute the location of the corner points
            bottom_left = [c1x - dy, c1y + dx]
            top_left = [c1x + dy, c1y - dx]
            bottom_right = [c2x - dy, c2y + dx]
            top_right = [c2x + dy, c2y - dx]

            corners = np.array([top_left, top_right, bottom_right, bottom_left] )
    
    # If no obstacles are detected, it is just an empty array
    else:
        corners = np.array([])
        centers = np.array([])

    return corners, centers, image_labels, obstacle_detected