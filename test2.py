from keras.datasets import cifar10

from datasets import Dataset
from utils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tSNE import *
from neuralREG_tsne import neuralREG_tSNE
from neural_tSNE import neural_tSNE
seed = 0

dataset = Dataset(seed)

X, y , X_train, y_train, X_test, y_test= dataset.get_MNIST_data(n_train=60000, n_test=10000)

regModel = neuralREG_tSNE()
regModel.load_model('Models/regularized/MNISTreg2RBM0.5')
Y = regModel.predict(X_test)
plot(Y, y_test, title='test regularized', s=1, linewidth=0.1, cmap='Paired')

parModel = neural_tSNE()
parModel.load_model('Models/parametric/MNISTpar2RBM')
Y = parModel.predict(X_test)
plot(Y, y_test, title='test parametric', s=1, linewidth=0.1,cmap='Paired')

lambdas = [0.5, 0.7, 0.9, 0.99]



