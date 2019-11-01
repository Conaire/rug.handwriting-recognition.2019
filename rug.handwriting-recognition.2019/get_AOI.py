#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: George Doulgeris (s3742237)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#AOI = Area of Interest
def get_AOI(direc):
    #read image to use it for contour finding
    img_rgb = cv2.imread(direc)
    
    
    

        
    #Convert image to HSV 
    img = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
    #Bilateral filtering (keeps edges sharp while removing noise)
    img = cv2.bilateralFilter(img,9,105,105)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret,thresh_image = cv2.threshold(img,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    #Detect Edges    
    canny_image = cv2.Canny(thresh_image,250,255)
    canny_image = cv2.convertScaleAbs(canny_image)
    kernel = np.ones((3,3), np.uint8)
    
    #Dilate to make edges sharper in order to get the contours easier
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    dilated_image = np.pad(dilated_image, ((0,0),(2,2)) , 'constant',constant_values=255)
#    cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
#    cv2.imshow("dilated",dilated_image);cv2.waitKey(0)
    #Find Contours and get the largest contour which encircles our Area of Interest
    #contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
   
    c=contours[0]
    
    
    
 
 
#    final = cv2.drawContours(img, [c], -1, (255,255, 0), 3)
#    cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
#    cv2.imshow("Final",final);cv2.waitKey(0)
    
    
    
    mask = np.ones(img_rgb.shape,np.uint8)
    
    new_image = cv2.drawContours(mask,[c], -1, (255,255,255), -1)
    alpha = np.array(new_image,copy=True)
#    new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
#    ret, thresh1 = cv2.threshold(new_image_gray,50,255 , cv2.THRESH_BINARY)
#    plt.imshow(thresh1);plt.title("Mask1");plt.show()

    
#    final = cv2.bitwise_and(img_rgb, img_rgb, mask = thresh1)
#    med = np.median(img)
#    print('Median of image is {}'.format(med))
#    final[final<=20] = med

#    final = cv2.bitwise_not(final)
        
    
    
    #Alpha blending
    backgr = np.full(img_rgb.shape, 255,dtype=np.uint8) #white background
    
    
    backgr = backgr.astype(float)
    foregr = img_rgb.astype(float)
    
    alpha = alpha.astype(float)/255
    
    #Check shapes to debug multiplication errors 
#    print("Shapes are {} and {}".format(alpha.shape,backgr.shape))    
    
    foregr = cv2.multiply(alpha,foregr)
    backgr = cv2.multiply(1.0 - alpha,backgr)
    
    outIm = cv2.add(foregr,backgr)
    outIm = outIm/255
    outIm = (outIm)*255
    outIm = outIm.astype(np.uint8)
    
    
    
    
    
    
    x,y,w,h = cv2.boundingRect(c)

    
#    new = outIm[y:y+h,x:x+w]
    
    
    
    
    

    
#    cv2.imshow("Final Result",new)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
# ####################################################################################

  
    

    
    
    return outIm,x,y,w,h
    
    


#if __name__ == '__main__':
#    direc = os.getcwd()
#    direc = os.path.join(direc,"data/P423-1-Fg002-R-C01-R01-fused.jpg")
#
#    src = cv2.imread(direc)
#    im = get_AOI(direc)
#
#
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,2,1)
#    ax1.set_title('Original Image')
#    ax1.imshow(src)
#    ax2 = fig.add_subplot(1,2,2)
#    ax2.set_title('Filtered image')
#    ax2.imshow(im)
#
#    plt.show()
