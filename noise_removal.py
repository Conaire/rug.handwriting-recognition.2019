import cv2
import numpy as np
from binarization import  *
import get_AOI as aoi

def closing(img):
    kernel = np.ones((5,5),np.uint8)
    print(">> Shape of image is: {0}".format(img.shape))
    print(">> Noise removal through closing with kernel: {0}".format(kernel))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def fast_nlmd(img):
    img_blur = cv2.bilateralFilter(img, 3, 75, 75)
    img_denoised = cv2.fastNlMeansDenoising(img_blur, templateWindowSize=7, searchWindowSize=21, h=15)
    return img_denoised

# to test just run the script
# binarization and closing
if __name__ == '__main__':
    img = aoi.get_AOI('data/P423-1-Fg002-R-C01-R01-fused.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clean_img = fast_nlmd(img)
    bin_img = changs_method(clean_img)

    plt.subplot(1, 2, 1), plt.imshow(clean_img, 'gray'), plt.title('Binary')
    plt.subplot(1, 2, 2), plt.imshow(bin_img, 'gray'), plt.title('Cleaned')
    plt.show()
