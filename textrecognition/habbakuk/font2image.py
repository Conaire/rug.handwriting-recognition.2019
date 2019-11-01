#Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
#Load the font and set the font size to 42
num_samples = 100
#Character mapping for each of the 27 tokens
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
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}

#Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')

    #Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)    
    draw = ImageDraw.Draw(img)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char_map[label], 0, font)

    return img

#Create a 50x50 image of the Alef token and save it to disk
#To get the raw data cast it to a numpy array
    
font = ImageFont.truetype('Habbakuk.TTF', 42)

#for letter in char_map:
#    path = letter
#    try:  
#        os.rmdir(path)
#    except OSError:  
#        print ("Deletion of the directory %s failed" % path)
#    else:  
#        print ("Successfully deleted the directory %s" % path)

for letter in char_map:
    path = letter
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)


font_size_list = [20, 25, 30, 35, 38, 40, 42, 45, 50, 55]
num_samples_per_size = num_samples/np.size(font_size_list)
index_size = 0

for letter in char_map:
    for count in range(0, num_samples):
        if count % num_samples_per_size == 0 and count != 0:
            index_size = index_size + 1
            
        font = ImageFont.truetype('Habbakuk.TTF', font_size_list[index_size])
        img = create_image(letter, (50, 56))
        img.save(letter + '/' + letter + '_' + str(count) + '.png')
    index_size = 0






