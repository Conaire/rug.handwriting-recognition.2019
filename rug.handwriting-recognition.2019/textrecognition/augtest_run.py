# set the matplotlib backend so figures can be saved in the background
import matplotlib
from keras.preprocessing.image import ImageDataGenerator

from textrecognition import HwAlexNet

from textrecognition.data import getdata
from textrecognition.recognition_conifg import recognition_config

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
args = recognition_config


data, labels = getdata(args["dataset"])


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

image = data[1]
cv2.imshow('image', image)
cv2.waitKey(0)

image = aug.apply_transform(image, {"zx": 2})


cv2.imshow('image', image)
cv2.waitKey(0)