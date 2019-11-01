from matplotlib import pyplot as plt
import cv2
from skimage.morphology import thin, skeletonize
import numpy as np
import math

def hough_skew_correct(in_img):
    # dimensions of the image
    rows, cols = in_img.shape
    # erosion removes the outline of the parchment
    eroded_image = cv2.erode(in_img, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    # form blocks by dilating the words
    dilated_image = cv2.dilate(eroded_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=8)
    closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # thin blocks to lines
    thinned_image = skeletonize(closed_image)
    closed_image[~thinned_image] = 0
    closed_image = cv2.dilate(closed_image, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    # detect lines
    lines = cv2.HoughLinesP(closed_image, rho=1, theta=np.pi / 180, threshold=200, minLineLength=100, maxLineGap=200)
    # will store line angles
    line_angles = []
    lined_img = np.ones(in_img.shape)
    for line in lines:
        for x1, y1, x2, y2 in line:
            # get line angle
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            line_angles.append(angle)
            cv2.line(lined_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    print(len(lines))
    # plt.subplot(1, 2, 1), plt.imshow(in_img, 'gray')
    # plt.subplot(1, 2, 2), plt.imshow(lined_img, 'gray'), plt.show()
    # check if skewness correction is necessary
    zero_angles = [a for a in line_angles if a == 0]
    if (len(zero_angles) < len(line_angles)):
        # discard horizontal lines (0 degrees)
        line_angles = [a for a in line_angles if a != 0]
        # discard lines with a slope larger than 45 degrees, as they are most probably misidentified.
        line_angles = [a for a in line_angles if abs(a) < 45]
        # find slope of majority and use their angle for skewness correction
        pos_angles = [a for a in line_angles if a > 0]
        neg_angles = [a for a in line_angles if a < 0]
        if len(pos_angles) > len(neg_angles):
            angles_to_use = pos_angles
        else:
            angles_to_use = neg_angles
        # rotation based on average angle of lines
        avg_angle = np.median(angles_to_use)
        rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), avg_angle, 1)
        # rotate image
        corrected_img = cv2.warpAffine(in_img, rotation, (cols, rows))
        # look at beautiful results
        # plt.subplot(1, 2, 1), plt.title('original'), plt.imshow(in_img, 'gray')
        # plt.grid(True)
        # plt.subplot(1, 2, 2), plt.title('corrected by {0}'.format(avg_angle)), plt.imshow(corrected_img, 'gray')
        # plt.grid(True)
        # plt.show()
        return corrected_img
    else:
        return in_img


