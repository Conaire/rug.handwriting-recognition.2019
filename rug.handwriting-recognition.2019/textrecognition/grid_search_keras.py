
# Use scikit-learn to grid search the learning rate and momentum
import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
from sklearn.preprocessing import LabelBinarizer

from recognition_data import recognition_data
from pre_train import get_pretrained_model


from textrecognition.recognition_conifg import *

#args = {
 #   #"dataset": "habbakuk/font_dataset",
  #  "dataset": "../monkbrill2",
   # "plot": "output_font/font_cnn_plot.png",
    #"model": "output_font/font_cnn_drop.h5",
    #"label_bin": "output_font/font_cnn_lb.pickle",
#}


def create_model(learn_rate=0.01, momentum=0):


    print("learning rate")
    print(learn_rate)

    model = get_pretrained_model(learn_rate, momentum)



    #model = HwAlexNet.HwAlexNet.build(width=56, height=56, depth=1, classes=27)

    return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables


realData, realLabels  = recognition_data["data"] #getdata(recognition_config["dataset"])
lb = LabelBinarizer()
realLabels = lb.fit_transform(realLabels)


#X = dataset[:,0:8]
#Y = dataset[:,8]



# create model
model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.1 , 0.2, 0.3]
momentum = [0.0, 0.2 , 0.4, 0.6, 0.8, 0.9]

param_grid = dict(learn_rate=learn_rate,  momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(realData, realLabels)




# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))