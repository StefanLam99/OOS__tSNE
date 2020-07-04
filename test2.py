from keras.datasets import cifar10

from datasets import Dataset
from utils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tSNE import *
from reg_tSNE import neuralREG_tSNE
from par_tSNE import neural_tSNE
seed = 0

dataset = Dataset(seed)
model_type = 'reg' # par/reg/auto/PCA/kernel
data_name = 'COIL20' # MNIST/ COIL20
d_components = 2
grad_method = 'SGD'

if data_name == 'COIL20':
    n_train = 960
    n_test = 480
    batch_size = 480
    im_shape = (32,32)
    color = 'nipy_spectral'
elif data_name == 'MNIST':
    n_train = 10000
    n_test = 5000
    batch_size = 1000
    im_shape = (28,28)
    color = 'Paired'

X, labels, X_train, labels_train, X_test, labels_test = dataset.get_data(data_name,n_train, n_test )

file_path = 'Models/tSNE/' + data_name + str(n_train) + 'dim' + str(d_components) + grad_method + 'Y2.csv'
Y_gains = np.genfromtxt(file_path, delimiter = ',')
plot(Y=Y_gains, labels=labels_train, s=1, linewidth=0.2, cmap=color, axis='off')

file_path = 'Models/tSNE/' + data_name + str(n_train) + 'dim' + str(d_components) + 'ADAM' + 'Y2.csv'
Y_adam = np.genfromtxt(file_path, delimiter = ',')
plot(Y=Y_adam, labels=labels_train, s=1, linewidth=0.2, cmap=color, title='', axis='off')

