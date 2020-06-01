#Simple class to make a linear regression model
#Author: Stefan Lam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tSNE import cond_probs, joint_average_P
from time import time
from utils import plot, pca
from datasets import Dataset
from pca_tSNE import PCA_tSNE
import sys
np.set_printoptions(threshold=sys.maxsize)

seed = 0
class Linear_tSNE():

    def __init__(self, constant = True, initial_dim = None):
        self.coefficients = None
        self.constant = constant
        self.initial_dim = initial_dim
        return

    def train(self, X, y):
        '''
        train a multivariate linear regression model
        X: (nxD)
        y: (nxd)
        '''
        print('training linear regression for tSNE')
        (n, D) = X.shape
        if(self.initial_dim is not None):
            X = pca(X, self.initial_dim)

        t0 = time()
        if(self.constant):
            X = np.concatenate((np.ones((n,1)), X.copy()), axis=1)
        # linear regression formula: beta = ((X'X)^-1)X'y
        print(X)
        print(X.shape)

        self.coefficients = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),(X.T)),y)

        print('training took %.2f seconds' %(time()-t0))
        return self.coefficients

    def predict(self, X):
        if self.coefficients is None:
            print('Model not trained!')
            return
        if self.constant:
            (n,D) = X.shape
            X = np.concatenate((np.ones((n, 1)), X.copy()), axis=1)
        return np.dot(X,self.coefficients)

    def load_coefficients(self, coefficients):
        self.coefficients = coefficients

if __name__ == '__main__':
    dataset = Dataset(0)
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=6000, n_test=3000)
    data_name = 'MNIST'
    d_components = 2
    Y = np.genfromtxt('results/' + data_name + '/ADAMY' + str(d_components) +'.csv', delimiter=',')
    print(Y.shape)
    linear_model = Linear_tSNE()
    #np.linalg.inv(np.dot(X_train.T,X_train))
    #X_train = pca(X_train.copy(), 30).real
    #beta = linear_model.train(X_train, Y)
    pca_model = PCA_tSNE(initial_dim=2)
    pca_model.train(X_train)
    YY = pca_model.predict(X_train)
    #YY, M = pca(X_train.copy(), 2)
    #YY = YY.real
    #YY = linear_model.predict(X_test)

    plot(YY, y_train, s=1, linewidth=0.1, cmap='Paired', title='linear pca train')
    YYY = pca_model.predict(X_test)
    plot(YYY, y_test, s=1, linewidth=0.1, cmap='Paired', title='linear pca test')