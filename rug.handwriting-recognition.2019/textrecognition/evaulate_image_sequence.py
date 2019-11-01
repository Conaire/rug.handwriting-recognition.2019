from keras.models import load_model
import argparse
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from textrecognition.recognition_data import recognition_data
from textrecognition.recognition_conifg import recognition_config
from textrecognition.viterbi import viterbi
from textrecognition.hebrewbigrams import hebrew_start_probs, hebrew_states, hebrew_bigrams



from textrecognition.data import getdata



#{
 #   "dataset": "../monkbrill2",
  #  "plot": "output/simple_nn_plot.png",
   # "model": "output/simple_nn.model",
    #"label_bin": "output/simple_nn_lb.pickle",
#}


data, labels = recognition_data["data"] #getdata(args["dataset"])
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)


print(testX)


print("[INFO] loading network and label binarizer...")
model = load_model(recognition_config["model"])
lb = pickle.loads(open(recognition_config["label_bin"], "rb").read())


totest = np.array([data[0], data[1], data[2]])

tester = np.ndarray(shape=(len(totest),56, 56, 3), buffer = totest)

# make a prediction on the image
preds = model.predict(tester, batch_size=32)

# find the class label index with the largest corresponding
# probability

print(preds)
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

print(lb.classes_)

labels = lb.classes_

results = []


for i in range(len(preds)):
   # print({ labels[li] : preds[i][li] for li in range(len(labels)) })
    results.append({ labels[li] : preds[i][li] for li in range(len(labels)) })


print(results)





states = hebrew_states


#for st in states:
 #   print(st)

start_p = hebrew_start_probs
trans_p = hebrew_bigrams
emit_p = lambda state, obs : obs[state]



viterbi(results, states, start_p, trans_p, emit_p)


