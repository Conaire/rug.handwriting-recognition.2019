import os, sys
import pandas as pd
import font2image as f2i
from PIL import Image
import numpy as np
import cv2


def read_ngram_file(filepath):
    os.makedirs("./ngram_images/", exist_ok=True)
    characterDF = pd.read_excel(filepath, sheet_name='new_list')
    for index, row in characterDF.iterrows():
        word = row['Names']
        heb_word = row['Hebrew_character']
        characters = word.split('_')
        characters.reverse()
        char_images = []
        for char in characters:
            char_img = f2i.create_image(char, (32, 32))
            char_images.append(char_img)
        widths, heights = zip(*(i.size for i in char_images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in char_images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save('./ngram_images/{}.png'.format(heb_word))
    return characterDF


# def generate_word(hebrew_word):
n_grams = read_ngram_file('ngrams.xlsx')