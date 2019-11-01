# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:01:42 2019

@author: George Doulgeris
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from binarization import *
from noise_removal import * 

def preprocessing(direc):
    """
    This function is all in one and combines all preprocessing methods that we 
    have developed so far to produce the final result to feed to the segmentation
    module. 
    
    It takes an image directory as input and produces a binarized image that 
    only contains the characters that we get from the binarization methods. 
    The whole process works like this:
        1) get_AOI: Takes an image and finds the largest contour. Use alpha
            blending in order to filter out unwanted objects. Outputs also include
            the coordinates of the bounding rectangle to be used for cropping later.
        2) fast_nlmd: Takes the RGB image from get_AOI and clears the noise
        3) changs_method: Binarizes the noise-free image using Chang's method
        4) preprocessing: Takes the binarized image from changs_method, finds 
            top ten largest contours which contain the holes. To filter out 
            contours that aren't encircle holes we figured (empirically)
            a heuristic measure to only get contours that are around holes.
            The main idea is that contours around holes have a large area but 
            the curve length is rather small, the opposite happens with contours
            that are mistakes, they tend to have large curve lengths but their 
            respective areas are small. That area/length ratio is what we use to 
            filter out the non-holes contours. The threshold was established
            empirically and seems to work perfectly. Finally, using the bounding
            rectangle coordinates from get_AOI we crop the image and serve it
            as output. 
            
        
    """
    src,x,y,w,h = get_AOI(direc)
    img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    denoised = fast_nlmd(img)
    binarized = changs_method(denoised)

    bina = np.array(binarized,copy=True)
    bina.dtype = 'uint8'
    fileName = direc.split('/')[2].split('.')[0]
    cv2.imwrite("binary/" + str(fileName) + ".png", bina.astype('uint8') * 255)

    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    legit_contours = []
    for c in contours:
#        print("Are of contour: {}, length of contour: {}, ratio: {}".format(cv2.contourArea(c), cv2.arcLength(c,True),cv2.contourArea(c)/cv2.arcLength(c,True)))
        area = cv2.contourArea(c)
        length= cv2.arcLength(c,True)
        if length != 0:
            rat = area/length
        if rat >= 9.5:
            legit_contours.append(c)
        
        
    bina = cv2.drawContours(binarized, legit_contours, -1, (0,0,0), thickness=cv2.FILLED)
    
    
    
    bina = bina[y:y+h,x:x+w]
    
    return bina
    
    
    
if __name__ == "__main__": 
    direc = os.getcwd()
    direc = os.path.join(direc,"data/P344-Fg001-R-C01-R01-fused.jpg")
    
    ready = preprocessing(direc)
    
    plt.imshow(ready,"gray");plt.title("Final preprocessing Result");plt.show()
    
