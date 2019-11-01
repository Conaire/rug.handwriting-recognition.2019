import operator

from textrecognition.recognition_trained_model import recognition_trained_model
import numpy as np
import cv2
from textrecognition.hebrewbigrams import hebrew_start_probs, hebrew_states, hebrew_bigrams
from textrecognition.viterbi import viterbi




def viterbi_for_word(letter_probabilities):
    states = hebrew_states

    # for st in states:
    #   print(st)

    start_p = hebrew_start_probs
    trans_p = hebrew_bigrams
    emit_p = lambda state, obs: obs[state]

    return viterbi(letter_probabilities, states, start_p, trans_p, emit_p)


def recognize_word(word_images):
    #images = np.array(word_images)

    #images = np.ndarray(shape=(len(images), 56, 56, 3), buffer=images)

    images = [pre_proc(img) for img in word_images]

    images = np.array(images, dtype="float") / 255.0

    preds = recognition_trained_model["model"].predict(images, batch_size=32)



    # labels = recognition_trained_model["model"].classes_
    labels = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem", "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial", "Waw", "Yod", "Zayin"]
    results = []

    for i in range(len(preds)):
        # print({ labels[li] : preds[i][li] for li in range(len(labels)) })
        results.append({labels[li]: preds[i][li] for li in range(len(labels))})

    #print(results)



    for result in results:


        predsTop = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=True)[:5])

        for key in result:
            if key not in predsTop:
                result[key] = 0



    return results





def pre_proc(image):
    image = np.stack((image,) * 3, axis=-1)

    h, w = image.shape[:2]
    h = min(h, 56)
    w = min(w, 56)

    image = cv2.resize(image, (w, h))

    blank_image = np.ones((56, 56, 3), np.uint8) * 255

    image = addImageToCenter(blank_image, image)

    return image


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






