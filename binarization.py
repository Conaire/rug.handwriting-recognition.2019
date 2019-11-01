#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Panagiotis Giagkoulas (s3423883)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from get_AOI import *

def global_binarization(img):
    """
    Method for global binarization using mean of image intensity as threshold
    :param img: grayscale image
    :return thres_img: binarized image
    """
    threshold = np.mean(img)
    print(">> Applying global binarization of image with threshold: {0}".format(threshold))
    _, thres = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY_INV)
    return thres


def otsus_binarization(img):
    """
    Method for otsu's binarization after applying gaussian blur
    :param img: grayscale image
    :return thres_img: binarized image
    """
    print(">> Applying Otsu's binarization of image after blurring with gaussian")
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # Gaussian blur seems to intensify bimodality of the image (necessary for otsu's)
    _, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thres


# function to get valid range for the yardstick
def get_yardstick_range(full_range, pos, rad):
    if pos - rad + 1 < 0:
        return slice(0, 2 * rad)
    elif pos + rad >= full_range:
        return slice(-2 * rad, full_range)
    else:
        return slice(pos - rad + 1, pos + rad + 1)


# function to get valid range for the local window
def get_window_range(full_range, pos, win_size):
    if pos - win_size//2 < 0:
        return slice(0, win_size)
    elif pos + win_size//2 >= full_range:
        return slice(-win_size, full_range)
    else:
        return slice(pos - win_size//2, pos + win_size//2 + 1)


def changs_method(img):
    """
    Method for chang's binarization method
    see: F. Chang, Retrieving information from document images: problems and solutions
    and: Gupta et al., OCR binarization and image pre-processing for searching historical documents
    :param img: grayscale image
    :return thres_img: binarized image
    """
    rows, cols = img.shape
    # mean and standard deviation on the image
    img_std = np.std(img)
    margin = 0.1
    print(">> Mean: {0}  Median: {1}  STD: {2}".format(np.mean(img), np.median(img), img_std))
    # choose global threshold
    if img_std/(np.mean(img)+0.1) < 0.2:
        # global threshold from otsu's
        ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # otsu's value but close to darker pixels
        ret = 255 - ret
    else:
        ret = 255 - np.mean(img)
    print(">> Mean is: {0}".format(np.mean(img)))
    print(">> Applying Chang's method with global threshold = {0}".format(ret))
    print(">> Local thresholding on pixel intensities: [{0} ,{1}]".format((ret - margin * img_std), (ret + margin * img_std)))
    # apply otsu's threshold on "far away" values
    thres_img = np.zeros(img.shape, dtype=np.uint8)
    thres_img[img >= (ret + margin * img_std)] = 0
    thres_img[img <= (ret - margin * img_std)] = 1
    # find and iterate over pixels with values close to global threshold
    indexes = np.array(np.nonzero(np.logical_and(img > (ret - margin * img_std), img < (ret + margin * img_std))))
    # range of local feature scale
    n_range = [2, 4, 6]
    # form yardsticks
    all_y_n = {}
    for n in n_range:
        all_y_n[n] = np.concatenate(
            (np.ones((n // 2,), dtype=int),
             -1 * np.ones((n,), dtype=int),
             np.ones((n // 2,), dtype=int)), axis=None)
    # to store res for each scale
    scale_res = {}
    # go over all chosen pixels 
    for i in range(indexes.shape[1]):
        # for different scale of local feature
        for n in n_range:
            # form neighbourhood of pixel
            x_p = indexes[0, i]
            y_p = indexes[1, i]
            # vertical neighbourhood
            v_range = get_yardstick_range(rows, x_p, n)
            v_n = np.array(img[v_range, y_p])
            # horizontal neighbourhood
            h_range = get_yardstick_range(cols, y_p, n)
            h_n = np.array(img[x_p, h_range])
            # store scale value
            res = max(np.dot(h_n, all_y_n[n]), 0) + max(np.dot(v_n, all_y_n[n]), 0)
            scale_res[n] = res
        # best scale + 1 is chosen as window size for local binarization
        win_size = max(scale_res, key=lambda key: scale_res[key]) + 1
        x_range = get_window_range(rows, x_p, win_size)
        y_range = get_window_range(cols, y_p, win_size)
        window = img[x_range, y_range]
        # find and apply local threshold for chosen pixel area
        local_threshold = np.mean(window)
        if thres_img[x_p, y_p] >= local_threshold:
            thres_img[x_p, y_p] = 0
        else:
            thres_img[x_p, y_p] = 1

    return thres_img


# to test just run the script
if __name__ == '__main__':
    ##    img = cv2.imread('data/P123-Fg002-R-C01-R01-fused.jpg') # using edited image!!!!
    ##    print(">> Shape of image is: {0}".format(img.shape))
    img = get_AOI('data/P123-Fg001-R-C01-R01-fused.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # needed for otsu's and adaptive thresholding
    bin_img = changs_method(img)

    #plt.subplot(2, 2, 1), plt.imshow(img, 'gray'), plt.title('Original')
    #plt.subplot(2, 2, 2), plt.imshow(otsus_binarization(img), 'gray'), plt.title('Otsu\'s')
    plt.imshow(bin_img, 'gray'), plt.title('Chang\'s')
    #plt.subplot(2, 2, 4), plt.imshow(global_binarization(img), 'gray'), plt.title('Global')
    plt.show()
