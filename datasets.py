# Author: Stefan Lam
# Preprocesses and loads several datasets

import numpy as np
from math import pi, cos
from sklearn.preprocessing import normalize, minmax_scale
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils import plot
from tSNE import *
class Dataset:

    def __init__(self, seed):
        self.seed = seed

# natural datasets

    def get_MNIST_data(self, n_train = 3000, n_test = 1000):
        '''
        Preprocess and load MNIST datasets, consisting of grayscale imaged of hand written
        integers from 0-9 made out of 28*28 pixels with RGB values up till 255.
        train set has 60000 samples
        test set has 10000 samples
        '''
        np.random.seed(self.seed)
        print('loading and preprocessing MNIST data...')
        data_train = np.genfromtxt('data/MNIST/mnist_train.csv', delimiter=',')
        X = data_train[:, 1:].astype(np.float32, copy=False)/255
        y = data_train[:,0]

        y_train = data_train[0:n_train, 0]
        X_train = data_train[0:n_train, 1:]
        X_train.astype(np.float32, copy=False)
        X_train = X_train/ 255 # RGB values are max 255

        data_test = np.genfromtxt('data/MNIST/mnist_test.csv', delimiter=',')
        y_test = data_test[0:n_test, 0]
        X_test = data_test[0:n_test, 1:]
        X_test.astype(np.float32, copy=False)
        X_test = X_test/ 255 # RGB values are max 255

        return X, y, X_train, y_train, X_test, y_test

    def get_CIFAR10_data(self, n_train = 3000, n_test = 1000):
        '''
        Preprocess and load CIFAR10 dataset, consisting of color images of 10 different objects
        integers from 0-9 made out of 28*28*3 pixels with RGB values up till 255.
        has 50000 samples, first column are the labels
        '''
        np.random.seed(self.seed)
        print('loading and preprocessing CIFAR10 data...')
        data_train = np.genfromtxt('data/CIFAR10/cifar10_data.csv', delimiter=',')
        X = data_train[:, 1:].astype(np.float32, copy=False)/255
        y = data_train[:,0]

        y_train = data_train[0:n_train, 0]
        X_train = X[0:n_train, 1:]


        data_test = np.genfromtxt('data/MNIST/mnist_test.csv', delimiter=',')
        y_test = data_test[0:n_test, 0]
        X_test = X[0:n_test, 1:]



        return X, y, X_train, y_train, X_test, y_test

    def get_LETTER_data(self, n_train = 6000, n_test = 1000):
        '''
        Preprocesses and loads the letter recognition data. 20000 samples describing
        20 differnt fonts of the classes 26 capital letters of the english alphabet.
        Features are statistical measures and edge counts
        '''
        np.random.seed(self.seed)
        print('loading and preprocessing LETTER data...')
        data = np.genfromtxt('data/LETTER/letter_data.csv', delimiter=',')
        np.random.shuffle(data)
        X = data[:, 1:].astype(np.float32, copy=False)
        X = minmax_scale(X)
        y = data[:,0]

        y_train = data[0:n_train, 0]
        X_train = data[0:n_train, 1:]

        y_test = data[n_train:n_train+n_test, -1].astype(np.uint32, copy = False)
        X_test = data[n_train:n_train+n_test, 0:-1].astype(np.float32, copy=False)

        return X, y, X_train, y_train, X_test, y_test

    def get_IRIS_data(self, n_train = 100, n_test = 50):
        '''
        Preprocess and load IRIS data consistin of three types of flowers
        stored in a 150x5 array with the first 4 colums the features:
        Sepal Length, Sepal Width, Petal Length and Petal Width.
        last column are the classes:
        0: 'setosa'
        1: 'versicolor'
        2: 'virginica'
        '''
        np.random.seed(self.seed)
        print('loading and preprocessing IRIS data...')
        data = np.genfromtxt('data/IRIS/iris_data.csv', delimiter=',')
        np.random.shuffle(data)
        X = data[:,0:-1]
        y = data[:,-1].astype(np.uint32, copy = False)

        features = data[:,0:-1]
        labels = data[:, -1].astype(np.uint32, copy = False)
        features = normalize(features)

        y_train = labels[0:n_train]
        X_train = features[0:n_train, 0:-1]

        y_test = labels[n_train:n_train+n_test]
        X_test = features[n_train:n_train+n_test, 0:-1]

        return X, y ,X_train, y_train, X_test, y_test


    def get_coil20_data(self):
        '''
        preproces and load coil20 data consisting of images of 20 different objects
        taken at 72 different angels. The images are 32x32 = 1024 pixels. The dataset
        is already normalized, and it contains 20*72 = 1440 samples. The last columns
        is the label (1-20) of the object .
        '''
        np.random.seed(self.seed)
        print('loading and preprocessing coil20 data...')
        data = np.genfromtxt('data/coil20/coil20_data.csv', delimiter=',')
        np.random.shuffle(data)
        X = data[:,0:-1].astype(np.float32, copy=False)
        y = data[:,-1].astype(np.uint32, copy = False)

        data_train = np.genfromtxt('data/coil20/coil20_train.csv', delimiter=',').astype(np.float32, copy = False)
        np.random.shuffle(data_train)
        y_train = data_train[:,-1]
        X_train = data_train[:, 0:-1]

        data_test = np.genfromtxt('data/coil20/coil20_test.csv', delimiter=',').astype(np.float32, copy = False)
        np.random.shuffle(data_test)
        y_test = data_test[:,-1]
        X_test = data_test[:, 0:-1]
        return X, y, X_train, y_train, X_test, y_test


# artificial datasets

    def get_SWISS_data(self):
        n = 5000
        np.random.seed(self.seed)
        p = np.random.uniform(size=(n, 1))
        q = np.random.uniform(size=(n, 1))

        t = ((3*pi)/2)*(1+2*p)
        print(t)
        X = np.zeros(shape=(n, 3))
        print((t*np.cos(t)).shape)
        X[:,0] = (t*np.sin(t)).reshape(n)
        X[:,1] = (30*q).reshape(n)
        X[:,2] = (t*np.cos(t)).reshape(n)

        X += 0.05*np.random.randn(n, 3)

        labels = np.remainder(np.sum(np.concatenate((np.round(t/2), np.round((30*q)/12)), axis=1),axis=1), 2)

        return X, labels




# helper function
    def get_data(self, name, n_train, n_test):
        if( name == 'MNIST'):
            return self.get_MNIST_data(n_train=n_train, n_test=n_test)
        elif(name == 'COIL20'):
            return self.get_coil20_data()
        elif(name == 'IRIS'):
            return self.get_IRIS_data(n_train=n_train, n_test=n_test)
        elif(name == 'LETTER'):
            return self.get_LETTER_data(n_train=n_train, n_test=n_test)
        elif(name == 'CIFAR10'):
            return self.get_CIFAR10_data(n_train=n_train, n_test=n_test)
        elif(name == 'SWISS'):
            return self.get_SWISS_data()
        elif(name == 'IRIS'):
            return self.get_IRIS_data(n_train=n_train, n_test=n_test)
        else:
            print('No dataset with that name!')


if __name__ == '__main__':
    dataset = Dataset(0)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    X, labels = dataset.get_SWISS_data()
    print(X)
    X = minmax_scale(X)
    print(X)
    print(labels.shape)
    print(X[:,2].shape)
    print(labels)
    print(X[:,2])
    ax.scatter3D(X[:,0], X[:,1], X[:,2], c = labels, cmap='Paired', s=2, linewidths=0.1)
    plt.show()
    model = tsne(random_state=0, initialization=None, grad_method='ADAM', perplexity=40,
                 max_iter=1000, learning_rate=0.01)

    Y, cost = model.transform(X)

    plot(Y, labels, cmap='Paired', s=1, linewidth=0.1)





