from keras.datasets import cifar10

from datasets import Dataset
from utils import *
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tSNE import *
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
''' 
X = np.genfromtxt('data/LETTER/LETTER.txt',delimiter=',', usecols=range(1,17))
labels = np.genfromtxt('data/LETTER/LETTER.txt',delimiter=',', usecols=[0], dtype=np.str)
print(X)
print(labels)
unique = np.unique(labels)
reallabels = np.zeros(len(labels))
print(len(labels))
for i, e in enumerate(labels):
    for j, u in enumerate(unique):
        if e==u:
            reallabels[i] = j
            break
print(reallabels)
reallabels = reallabels + 1
reallabels = reallabels.astype(np.int32)
print(np.unique(reallabels))
print(np.unique(labels))
print(X.shape)
reallabels = reallabels.reshape(20000,1)
X = np.concatenate((reallabels, X), axis=1)
print(X)

np.savetxt('data/LETTER/letter_data.csv',X, delimiter=',')
'''

alphabet = ['A', 'B', 'C' ,'D' ,'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z']
seed = 0
dataset = Dataset(seed)
#X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=6000)
y_train = np.genfromtxt('results/MNIST/labels6000train0.csv', delimiter=',')
cmap = cm.get_cmap('gist_earth', 26)
print(type(cmap))

model = tsne(random_state=0, initialization='no', initial_dims=100, grad_method='gains', perplexity=40, max_iter=1000, data_name='LETTER', learning_rate=500)
#Y = model.transform(X_train)
Y = np.genfromtxt('results/MNIST/ADAMY2.csv', delimiter=',')
marker=['0', '1','2','3','4','5','6','7','8','9']
plot(Y, y_train, label=True, s=1, linewidth=0.1, save_path='results/MNIST/Plots/scatter')



''' 
print(X_train.shape)
X_train = X_train.reshape(50000,1024,3)
print(X_train.shape)
X_train = np.mean(X_train, axis=2)
print(X_train.shape)
n_samples = 1000
a= X_train.reshape(50000,-1).astype(np.float32)[0:n_samples,:]
a=a/255
y = y_train[0:n_samples].reshape(n_samples).astype(np.int32)
print(X_train)
print(a.shape)
model = tsne(random_state=0, initialization='PCA', initial_dims=100, grad_method='gains', perplexity=40, max_iter=1000, data_name='CIFAR10', learning_rate=500)
Y = model.transform(a)
print(y)
#Y = np.genfromtxt('results/CIFAR10/gainsY2.csv', delimiter=',')
plot(Y, y)
print(y_train.shape)
b = np.concatenate((y_train, a),axis=1).astype(np.int32)
print(b.shape)
print(np.unique(y_train))
#np.savetxt('data/CIFAR10/cifar10_data.csv',b,delimiter=',')
t0 = time()
#a = np.genfromtxt('data/CIFAR10/cifar10_data.csv',delimiter=',')
print(a.shape)
print(time()-t0)
'''