from math import ceil

import matplotlib
from keras.preprocessing.image import ImageDataGenerator


matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


def getdata(dataset):
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    # imagePaths = sorted(list(paths.list_images(args["dataset"])))
    imagePaths = sorted(list(paths.list_images(dataset)))

    print(imagePaths)

    # random.shuffle(imagePaths)

    # imagePaths = imagePaths[0:120]
    random.seed(42)
    random.shuffle(imagePaths)
    #imagePaths = imagePaths[0:2000]

    #print(imagePaths)


    i = 0

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize it to 64x64 pixels (the required input
        # spatial dimensions of SmallVGGNet), and store the image in the
        # data list

        print(imagePath)

        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        #image = cv2.resize(image, (56, 56))
        #image = np.expand_dims(image, axis=2)
        image = np.stack((image,) * 3, axis=-1)

        h, w = image.shape[:2]
        h = min(h, 56)
        w = min(w, 56)

        image = cv2.resize(image, (w, h))

        blank_image = np.ones((56, 56, 3), np.uint8) * 255

        image = addImageToCenter(blank_image, image)

        data.append(image)

        #height, width, c = image.shape


        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)


        #print(label);

        if i % 1000 == 0:
            print("so far," + str(i) + " files read ")

        i += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels



def findCenter(img):

    h, w, c = img.shape
    return int(w / 2), int(h / 2)

def addImageToCenter(img1, img2):

    pt1 = findCenter(img1)
    pt2 = findCenter(img2)

    ## (2) Calc offset
    dx = (pt1[0] - pt2[0])
    dy = (pt1[1] - pt2[1])

    h, w = img2.shape[:2]


    dst = img1.copy()
    dst[dy: dy + h, dx: dx + w] = img2

    return dst



