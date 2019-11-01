# set the matplotlib backend so figures can be saved in the background
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
import numpy

from textrecognition import HwAlexNet

from textrecognition.data import getdata

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
from textrecognition.recognition_conifg import *


args = recognition_config

#args = {
 #   "dataset": "../monkbrill2",
  #  "plot": "output/simple_nn_plot.png",
   # "model": "output/simple_nn.model",
    #"label_bin": "output/simple_nn_lb.pickle",
#}

print("[INFO] loading images...")



data_aug_type = "zoom_range"
data_aug_start = 0.0
data_aug_stop = 1
data_aug_step = 0.1

# grab the image paths and randomly shuffle them
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
data, labels = getdata(args["dataset"])

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


print(labels)

model = HwAlexNet.HwAlexNet.build(width=56, height=56, depth=1,
                                  classes=len(lb.classes_))


accs = []

for counter in numpy.arange(data_aug_start, data_aug_stop + data_aug_step, data_aug_step):


    print(counter)

    #construct the image generator for data augmentation
    aug = ImageDataGenerator(**{ data_aug_type: counter, "fill_mode" :"nearest"})


    # initialize our VGG-like Convolutional Neural Network


    # initialize our initial learning rate, # of epochs to train for,
    # and batch size
    INIT_LR=0.01
    EPOCHS = 2
    BS = 32

    # initialize the model and optimizer (you'll want to use
    # binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,  metrics=["accuracy"])

    # train the network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                         epochs=EPOCHS)


    #H = model.fit(trainX, trainY, validation_data=(testX, testY),
      #  steps_per_epoch=len(trainX) // BS,
     #   epochs=EPOCHS, batch_size=32)


    results = model.evaluate(testX, testY)
    print(results)

    # evaluate the network
    print("[INFO] evaluating network...")
    #predictions = model.predict(testX, batch_size=32)
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))


    accs.append((counter, results[1]))




plt.style.use("ggplot")
plt.figure()

plt.xticks(numpy.arange(data_aug_start, data_aug_stop + data_aug_step, data_aug_step))
plt.plot([i[0] for i in accs], [i[1] for i in accs])

plt.title(data_aug_type)
plt.xlabel("Value")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print(accs)



