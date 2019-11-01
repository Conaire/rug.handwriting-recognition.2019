from keras.preprocessing.image import ImageDataGenerator

import HwAlexNet



import matplotlib

from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD, rmsprop
from pycm import ConfusionMatrix
import tensorflow as tf

from recognition_data import recognition_data
from data import getdata
from recognition_conifg import recognition_config

#args = recognition_config

#{
 #   "dataset": "habbakuk/font_dataset",
# #   "dataset": "monkbrill2",
   # "plot": "output_font/font_cnn_plot.png",
    #"model": "output_font/font_cnn_drop.h5",
   # "label_bin": "output_font/font_cnn_lb.pickle",
#}



def focal_loss_fixed(y_true, y_pred):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    gamma = 1.
    alpha = 0.99
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


INIT_LR=0.01
EPOCHS = 50
BS = 32

def get_pretrained_model(learn_rate=0.01, momentum = 0):


    print("pretrained")
    print(learn_rate)

    opt = SGD(lr=learn_rate, decay=1e-6, momentum = momentum )

    model = textrecognition.HwAlexNet.HwAlexNet.build(width=56, height=56, depth=1, classes=27)
    model.compile(loss=focal_loss_fixed, optimizer=opt, metrics=["accuracy"])
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    realData, realLabels = recognition_data["pretrain_data"] # getdata(args.pretrain_dataset)

    lb = LabelBinarizer()
    realLabels = lb.fit_transform(realLabels)

    model.fit(realData, realLabels,
              epochs=1, batch_size=32)


    return model






#model.compile(loss=focal_loss_fixed, optimizer=opt,  metrics=["accuracy"])