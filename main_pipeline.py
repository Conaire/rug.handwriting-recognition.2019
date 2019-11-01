# python main_pipeline.py -f ./data/grey-images-only/

import argparse
import segmentation as sg
import preprocessing_final as pp
import os, sys
import skewness_correction as skc
from os.path import isfile, join

from matplotlib import pyplot as plt
import cv2
from skimage.morphology import thin, skeletonize
import numpy as np
import math

# construct the argument parser and parse the arguments
from textrecognition.document_converter import convert_to_text, save_document
from textrecognition.word_recognizer import recognize_word, viterbi_for_word

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to the folder containing the 20 test images")
ap.add_argument("-w", "--words", required=False, help="path to the words output folder", default=0)
args = vars(ap.parse_args())

# iterate over images
# args['folder'] = './data/grey-images-only'
# args['words'] = './segments/words/'
imageList = os.listdir(args['folder'])
wordsFolder = args['words']
cls_imgs = []
lined_imgs = []
# imageList = os.listdir('./data/grey-images-only')
for image in imageList:
    if not isfile(join(args['folder'], image)):
        continue

    print('processing image: {}'.format(image))
    fileName = os.path.splitext(image)[0]
    # Area of Interest and Cleaning
    cleaned_image = pp.preprocessing(args['folder'] + '/' + image)
    # skewness correction
    try:
        corrected_image = skc.hough_skew_correct(cleaned_image)
    except TypeError:
        print(">> No lines detected for image: {0}. No correction took place.".format(image))
        corrected_image = cleaned_image

    try:
        # Segmentation
        document_images = sg.segment_image(corrected_image, fileName, wordsFolder)

        # recognition
        recognized_document = [[recognize_word(word) for word in line] for line in document_images]
        viterbi_corrected_document = [[viterbi_for_word(word) for word in line] for line in recognized_document]

        document_text = convert_to_text(viterbi_corrected_document)

        # saving output
        print(">> Saving output to {}/{}/{}.txt".format(args['folder'], "output", image))
        save_document(document_text, image, args['folder'])
    except:
        print("Error segmenting image {}".format(image))
