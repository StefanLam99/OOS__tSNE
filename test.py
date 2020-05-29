
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
data_name = 'COIL20'
file_path = 'results/'+data_name + '/' + data_name + 'costplot'
dataset = Dataset(seed)
#X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
Csgd = np.genfromtxt('results/'+data_name+'/SGDcost2.csv', delimiter=',')
Cadam = np.genfromtxt('results/'+data_name+'/ADAMcost2.csv', delimiter=',')
Cgains = np.genfromtxt('results/'+data_name+'/gainscost2.csv', delimiter=',')
start = 101
end = 1000
x = range(start,end)
plt.plot(x, Csgd[start:end], c='r')
plt.plot(x, Cgains[start:end], c='orange')
plt.plot(x, Cadam[start:end], c='b')

plt.xticks(np.arange(0,1001,100))
plt.xlim(0,1000)

plt.xlabel('Iteration')
plt.ylabel('t-SNE cost')
plt.legend(['SGD','a-SGD with momentum','ADAM' ], loc='upper right',frameon=True, fancybox=True)
plt.savefig(file_path)
plt.show()
