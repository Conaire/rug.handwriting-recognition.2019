#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:33:41 2019

@author: doulgeo
"""

import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


path = os.path.join(os.getcwd(),'ngram_images')

char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}',
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c',
            'Tsadi': 'c',
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}
            
hebrew  = ["א","ב","ג","ד","ה","ו","ז","ח","ט","י","ך","כ","ל","ם","מ","ן","נ","ס","ע","ף","פ","ץ","צ","ק","ר","ש","ת"]
#add zayin to the hebrew
hebrew.append("ז")

heb_eng = list(char_map.keys())

mapping = dict(zip(heb_eng,hebrew))    
filenames = os.listdir(path)

filtered = []
#Filter our numbers and underscores and get only labels
for filename in filenames:
    file,ext = os.path.splitext(filename)
    result = ''.join([i for i in file if not i.isdigit()])
    result = result.split("_")
    result.remove('')
    filtered.append(result)
  
#Map letters names to hebrew
final = []
for entry in filtered:
    mini_lis = []
    for i in range(len(entry)):
        if entry[i] in list(mapping.keys()):
         temp    = mapping[entry[i]]
        mini_lis.append(temp)
    final.append(mini_lis)
            


            
#Write labels to file 
with open('labels_eng.txt','w') as f:
    for item in filtered:
        f.write("{}\n".format(item))
        
with open('labels_heb.txt','w') as f2:
    for item in final:
        f2.write("{}\n".format(item))
    
    
#test
for i in range(3):
    impath = os.path.join(os.getcwd(),'ngram_images',filenames[i])
    img = mpimg.imread(impath)    
    plt.imshow(img)
    plt.title("Label: {}".format(filtered[i]))
    plt.show()
