'''
class to create a kernel t-SNE model
'''
import numpy as np
from time import time
from utils import norm_gauss_kernel,determine_sigma, cond_probs, pca, joint_Q, joint_average_P, profile, plot, distance_matrix, distance_matrix_squared
from datasets import Dataset
from tSNE import *

class kernel_tSNE:
    def __init__(self, d_components=2, initial_dims=30,X_train = None, random_state = 0, y_train = None,c=0.5):

        self.d_components = d_components # dimension of projection space
        self.initial_dims = initial_dims # initial dimensions of data, before applying t-sne
        self.X_train = X_train
        self.y_train = y_train
        self.c = c
        D = distance_matrix(self.X_train)
        self.sigma = determine_sigma(D, self.c)
        self.random_state = random_state
        self.trained = False


    @profile
    def train(self):
        '''
        train by implementing t sne to obtain y_train
        '''

        if (self.X_train == None):
            print('load data first')
            return
        X_train = self.X_train.copy()

        K_train = norm_gauss_kernel(X_train, X_train, self.sigma) # gram matrix between train data
        K_inverse = np.linalg.pinv(K_train) # pseudo inverse
        self.A = np.dot(K_inverse, self.y_train)
        self.trained = True

        return self.A


    def predict(self, X):
        '''
        predicts projections using kernel t-sne
        '''

        X_test = X
        X_train = self.X_train.copy()
        if self.trained == False:
            self.A = self.train()

        K_test = norm_gauss_kernel(X_train, X_test, self.sigma) # gram matrix between train and test data
        Y = np.dot(K_test, self.A)

        return Y

    def load_A(self, file_path):
        '''
        Load the alpha parameters
        '''
        self.A = np.genfromtxt(file_path, delimiter='')
        self.trained = True

    def save(self, file_path):
        if self.A == None:
            print('Train kernel t-SNE first!')
        np.savetxt(file_path, self.A, delimiter=',')


    def load(self, file_path):
        '''
        Load y_train from file_path
        '''
        self.y_train = np.genfromtxt(file_path, delimiter=',')

    def load_data(self, X_train):
        self.X_train = X_train





