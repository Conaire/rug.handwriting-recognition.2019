import cv2
import numpy as np
import pandas as pd
import os
from scipy.signal import argrelextrema, savgol_filter
from skimage.morphology import skeletonize

MIN_HEIGHT = 20
MIN_WIDTH = 20
MIN_CHAR_WIDTH = 20
MAX_CHAR_WIDTH = 48
VERTICAL_SMOOTHING = 20
HORIZONTAL_SMOOTHING = 51
LINE_HEIGHT_BUFFER_PERCENT = 0.6    # 70%


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def skeleton_image(image):
    # perform skeletonization
    skeleton = skeletonize(image)
    return skeleton


def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    a = a[np.logical_not(np.isnan(a))]
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


def watershedSplitting(chars_image, appendToList=[], heightThreshold=0):
    # ignoring 20% of horizontal borders
    # widthIgnore = int(np.round(0.2 * chars_image.shape[1]));
    # chars_image = chars_image[:, widthIgnore:chars_image.shape[1]-widthIgnore]

    if chars_image.shape[1] >= (MIN_CHAR_WIDTH*2):
        waterDrops = np.zeros(chars_image.shape[1])
        for index in range(chars_image.shape[1]):
            n = 0
            y = chars_image[n,index]
            while y == 0:
                waterDrops[index] += 1
                n += 1
                y = chars_image[n, index]
        # optimalCutoffPoint = np.argmax(waterDrops)
        optimalCutoffPoint = np.argmax(waterDrops[MIN_CHAR_WIDTH:len(waterDrops)-MIN_CHAR_WIDTH])
        optimalCutoffPoint += MIN_CHAR_WIDTH
        if waterDrops[optimalCutoffPoint] < heightThreshold:
            appendToList.append(chars_image)
            return appendToList
        else:
            cutoffHeightThreshold = waterDrops[optimalCutoffPoint] - 10
            prevPossibleCutoff = optimalCutoffPoint - MIN_CHAR_WIDTH
            nextPossibleCutoff = optimalCutoffPoint + MIN_CHAR_WIDTH
            if prevPossibleCutoff > MIN_CHAR_WIDTH and any(i >= cutoffHeightThreshold for i in waterDrops[0:prevPossibleCutoff+1]):
                # watershedSplitting(chars_image[:, 0:prevPossibleCutoff], appendToList, cutoffHeightThreshold)
                watershedSplitting(chars_image[:, 0:optimalCutoffPoint], appendToList, cutoffHeightThreshold)
            else:
                appendToList.append(chars_image[:, 0:optimalCutoffPoint])
            if nextPossibleCutoff < (chars_image.shape[1] - MIN_CHAR_WIDTH) and any(i >= cutoffHeightThreshold for i in waterDrops[nextPossibleCutoff:chars_image.shape[1]]):
                # watershedSplitting(chars_image[:, nextPossibleCutoff:chars_image.shape[1]], appendToList, cutoffHeightThreshold)
                watershedSplitting(chars_image[:, optimalCutoffPoint:chars_image.shape[1]], appendToList,
                                   cutoffHeightThreshold)
            else:
                appendToList.append(chars_image[:, optimalCutoffPoint:chars_image.shape[1]])
            return appendToList
    else:
        appendToList.append(chars_image)
        return appendToList


def segment_chars(word_image):
    word_image = cv2.threshold(word_image, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(word_image, connectivity=4)
    statsDF = pd.DataFrame(stats)

    # iterate through the connected components
    charSegmentsDF = []
    charactersDF = statsDF[statsDF[2]<word_image.shape[1]]
    for index, row in charactersDF.iterrows():
        # normal pre-segmented characters
        if row[2] >= MIN_CHAR_WIDTH and row[2]<= MAX_CHAR_WIDTH:
            charSegmentsDF.append(word_image[:, row[0]:row[0] + row[2]])
        # connected characters needing segmentation
        elif row[2] > MAX_CHAR_WIDTH:
            connected_char = word_image[:, row[0]:row[0] + row[2]]
            watershedSplitting(connected_char, charSegmentsDF)

    return charSegmentsDF


def segment_image(image_clean, filename, wordsFolder):
    # output directory
    # os.makedirs("./segments/" + filename, exist_ok=True)
    # os.makedirs("./rotations/", exist_ok=True)
    if wordsFolder != 0:
        os.makedirs(wordsFolder + filename, exist_ok=True)

    # opening - remove borders
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image_clean = cv2.morphologyEx(image_clean, cv2.MORPH_OPEN, kernel_open)

    # horizontal projection
    projection_h = np.sum(image_clean, 1)
    # projection_h = smooth(projection_h, HORIZONTAL_SMOOTHING)
    projection_h = savgol_filter(projection_h, 51, 3)
    peaks = argrelextrema(projection_h, np.greater)[0]

    # calculate average height of lines
    line_heights = []
    for peak_count in range(len(peaks)):
        if peak_count != 0:
            line_heights.append(peaks[peak_count] - peaks[peak_count-1])
    average_line_height = np.average(line_heights)
    line_lookup_area = LINE_HEIGHT_BUFFER_PERCENT * average_line_height

    # iterate through line segments
    line_count = 0

    # a list of lines containing list of images, representing words
    document_images = []

    for i in range(len(peaks)):
        peak = peaks[i]
        line_start_index = int(np.round(peak-line_lookup_area))
        line_stop_index = int(np.round(peak+line_lookup_area))
        if line_start_index < 0:
            line_start_index = 0
        if line_stop_index > image_clean.shape[0]:
            line_stop_index = image_clean.shape[0]
        line_segment = image_clean[line_start_index:line_stop_index, :]
        # dilation - find words
        kernel = np.ones((1, 10), np.uint8)
        img_dilation = cv2.dilate(line_segment, kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(np.uint8(img_dilation.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # check if words are found
        if len(contours) >= 2:
            line_count += 1

            current_line = []

            # sort contours
            (sorted_contours, boundingBoxes) = sort_contours(contours, method="right-to-left")

            word_count = 0
            for j, ctr in enumerate(sorted_contours):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(ctr)

                if h >= MIN_HEIGHT and w >= MIN_WIDTH:
                    word_count += 1
                    current_word = []
                    word = line_segment[y:y + h, x:x + w]
                    if wordsFolder != 0:
                        resized_word = cv2.resize(word, (128, 32), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(
                            wordsFolder + '{}/line_{}_word_{}.jpg'.format(filename, line_count, word_count),
                            cv2.bitwise_not(resized_word.astype('uint8') * 255))
                    characterSegmentsDF = segment_chars(word.astype('uint8') * 255)
                    # reverse characters
                    characterSegmentsDF.reverse()

                    for index in range(len(characterSegmentsDF)):
                        # find percentage of character in whole image
                        total_vol = characterSegmentsDF[index].shape[0] * characterSegmentsDF[index].shape[1]
                        character_vol = len(characterSegmentsDF[index][characterSegmentsDF[index] == 255])
                        percentage = int((character_vol / total_vol) * 100)

                        current_word.append(characterSegmentsDF[index])

                        # Save result
                        # cv2.imwrite('./segments/{}/line_{}_word_{}_char_{}.jpg'.format(filename, line_count, word_count, index),
                        #             cv2.bitwise_not(characterSegmentsDF[index]))

                    if len(current_word) > 0:
                        current_line.append(current_word)

            if len(current_line) > 0:
                document_images.append(current_line)

    return document_images
