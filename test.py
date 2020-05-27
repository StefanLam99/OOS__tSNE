
from keras.datasets import mnist
from keras import backend as K
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import Adam
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
seed = 0
dataset = Dataset(seed)
X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
Cadam = np.genfromtxt('results/COIL20/ADAMcost2.csv', delimiter=',')
Cgains = np.genfromtxt('results/COIL20/gainscost2.csv', delimiter=',')
x = range(1,1001)

plt.plot(x, Cadam)
plt.plot(x, Cgains)
plt.legend(['ADAM', 'gains'], loc='upper right')
plt.show()
