from matplotlib import pyplot as plt
import numpy as np
import keras
from tqdm import tqdm
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.datasets import mnist
from math import ceil
from keras.losses import mse
import keras.losses
from utils import *
from autoencoder import Autoencoder
class neural_tSNE:
    def __init__(self, d_components=2, perplexity=40., epochs=100, lr=0.01, random_state=0, batch_size=100, model=None, labda=0.99):
        #keras.losses.kl_loss = self.kl_loss # otherwise keras won't recognize this loss function
        self.d_components = d_components
        self.perplexity = perplexity
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.batch_size = batch_size
        self.model = model
        self.labda = labda


    def build_nn(self, n_input, layer_sizes = np.array([500, 500, 2000]), activations= np.array(['sigmoid','sigmoid','linear'])):
        ''''
        builds the structure for a multilayer feedforward neural network
        '''
        if self.model is not None:
            self.model = None
            print('Deleting current model for new model...')

        self.model = Sequential()
        #self.model.add(Dropout(0.5, input_shape=(n_input,)))
        self.model.add(Dense(layer_sizes[0], input_dim=n_input, activation=activations[0])) # input layer
        for size, activation in zip(layer_sizes[1:], activations[1:]): # hidden layers
            self.model.add(Dense(size, activation=activation))
        self.model.add(Dense(self.d_components))  # output layer
        self.model.compile(loss=self.kl_loss, optimizer=Adam(lr=self.lr))


    def train(self, X_train):
        """
        Trains the neural network
        """
        if self.model is None:
            print("We have no model!")
            return
        print('Start training the neural network...')

        begin = time()
        losses = np.zeros(self.epochs)
        n_sample, n_feature = X_train.shape
        nBatches = int(ceil(n_sample/self.batch_size))
        for epoch in range(self.epochs):
            new_indices = np.random.permutation(n_sample) # shuffle data for new random batches
            X = X_train[new_indices]
            loss = 0
            for i in range(nBatches):

                batch = X[i*self.batch_size:(i+1)*self.batch_size]
                blockPrint()

                cond_p, _ = cond_probs(batch.copy(), perplexity=self.perplexity)
                P = joint_average_P(cond_p)
                enablePrint()

                loss += self.model.train_on_batch(batch, P)

            losses[epoch] = loss/nBatches
            print('Epoch: %.d loss: %.3f elapsed time: %.2f' % (
            epoch + 1, losses[epoch], time() - begin))

        return losses
    def predict(self, X):
        """
        Makes prediction for a given data set X nxD
        """
        if self.model == None:
            print("Train the model first!")
            return
        return self.model.predict(X)

    def load_model(self, model):
        self.model = model
        self.model.compile(loss=self.kl_loss, optimizer=Adam(self.lr))


    def kl_loss(self,P, Y):
        # calculate neighbor distribution Q (t-distribution) from Y
        d = self.d_components
        dof = d - 1.  # degrees of freedom for student t distribution
        n = self.batch_size
        eps = K.variable(10e-15) # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)

        sum_act = K.sum(K.square(Y), axis=1)
        Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(Y, K.transpose(Y))
        Q = (sum_act + Q) / dof
        Q = K.pow(1 + Q, -(dof + 1) / 2)

        # delete diagonals
        Q *= K.variable(1 - np.eye(n))

        #normalize
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)

        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)

        return C

    def save(self, file_path = None):
        if file_path==None:
            print("No file path specified!")
            return
        self.model.save(file_path)

