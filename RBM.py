'''
    Adapted from the MatLab code by Ruslan Salakhutdinov and Geoff Hinton
    Avilable at: http://science.sciencemag.org/content/suppl/2006/08/04/313.5786.504.DC1
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from utils import profile
learning_rate = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RBM:
    '''
    A class that trains a RBM
    '''
    def __init__(self, v_dim, h_dim):
        '''
            v_dim = dimension of the visible layer
            h_dim = dimension of the hidden layer
        '''
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.W = np.zeros((self.v_dim, self.h_dim))
        self.a = np.zeros((self.v_dim, 1))
        self.b = np.zeros((self.h_dim, 1))
        return

    @classmethod
    def from_Values(cls, weights):
        '''
            Initialize with trained weights.
        '''
        W, a, b = weights['W'], weights['a'], weights['b']
        assert (W.shape[0] == a.shape[0]) and (W.shape[1] == b.shape[0])
        rbm = cls(W.shape[0], W.shape[1])
        rbm.W = W
        rbm.a = a
        rbm.b = b
        return rbm

    @classmethod
    def from_File(cls, filename):
        '''
            Initialize with weights loaded from a file.
        '''
        return cls.from_Values(RBM.load_weights(filename))

    def v_probs(self, h):
        '''
            Input:
            - h has shape (h_dim,m)
            - a has shape (v_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert (h.shape[0] == self.h_dim)
        v_probs = sigmoid(self.a + np.dot(self.W, h))
        assert (not np.sum(np.isnan(v_probs)))
        return v_probs

    def h_probs(self, v):
        '''
            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert (v.shape[0] == self.v_dim)
        h_probs = sigmoid(self.b + np.dot(self.W.T, v))
        assert (not np.sum(np.isnan(h_probs)))
        return h_probs

    @profile
    def train(self, x, epochs=20, batch_size=100, learning_rate=learning_rate, plot=False, initialize_weights=True):
        '''
            Trains the RBM with the 1-step Contrastive Divergence algorithm (Hinton, 2002).

            Input:
            - x has shape (v_dim, number_of_examples)
            - plot = True plots debugging related plots after every epoch
            - initialize_weights = False to continue training a model
              (e.g. loaded from earlier trained weights)
        '''
        assert (x.shape[0] == self.v_dim)

        np.random.seed(0)

        # track mse
        error = 0.
        error_sum = 0.

        # hyperparameters used by Hinton for MNIST
        initialmomentum = 0.5
        finalmomentum = 0.9
        weightcost = 0.0002
        num_minibatches = int(x.shape[1] / batch_size)

        DW = np.zeros((self.v_dim, self.h_dim))
        Da = np.zeros((self.v_dim, 1))
        Db = np.zeros((self.h_dim, 1))
        t0 = time()
        # initialize weights and parameters
        if initialize_weights == True:
            self.W = np.random.normal(0., 0.1, size=(self.v_dim, self.h_dim))
            self.a = np.zeros((self.v_dim, 1))
            self.b = np.zeros((self.h_dim, 1))

        for i in range(epochs):

            #np.random.shuffle(x.T)

            if i > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum

            for j in range(num_minibatches):
                # get the next batch
                v_pos_states = x[:, j * batch_size:(j + 1) * batch_size]

                # get hidden probs, positive product, and sample hidden states
                h_pos_probs = self.h_probs(v_pos_states)
                pos_prods = v_pos_states[:, np.newaxis, :] * h_pos_probs[np.newaxis, :, :]
                h_pos_states = np.random.binomial(1, h_pos_probs) # gibbs sampling step

                # get negative probs and product
                v_neg_probs = self.v_probs(h_pos_states)
                h_neg_probs = self.h_probs(v_neg_probs)
                neg_prods = v_neg_probs[:, np.newaxis, :] * h_neg_probs[np.newaxis, :, :]

                # compute the gradients, averaged over minibatch, with momentum and regularization
                cd = np.mean(pos_prods - neg_prods, axis=2)
                DW = momentum * DW + learning_rate * (cd - weightcost * self.W)
                Da = momentum * Da + learning_rate * np.mean(v_pos_states - v_neg_probs, axis=1, keepdims=True)
                Db = momentum * Db + learning_rate * np.mean(h_pos_probs - h_neg_probs, axis=1, keepdims=True)

                # update weights and biases
                self.W = self.W + DW
                self.a = self.a + Da
                self.b = self.b + Db

                # log the mse of the reconstructed images
                error = np.mean((v_pos_states - v_neg_probs) ** 2)
                error_sum = error_sum + error

            print("Epoch: %d Reconstruction MSE: %.4f elapsed time: %.2f" % (i + 1,error_sum, time()-t0))
            error_sum = 0.

        return self.h_probs(x)

    def save(self, filename):
        '''
            Save trained weights
        '''
        weights = {"W": self.W, "a": self.a, "b": self.b}
        RBM.save_weights(weights, filename)
        return

    @staticmethod
    def save_weights(weights, filename):
        '''
            Save RBM weights to file
        '''
        np.savetxt(filename + '_a.csv', weights['a'], delimiter=",")
        np.savetxt(filename + '_b.csv', weights['b'], delimiter=",")
        np.savetxt(filename + '_W.csv', weights['W'], delimiter=",")
        return

    @staticmethod
    def load_weights(filename):
        '''
            Save RBM weights to file
        '''
        W = np.loadtxt(filename + '_W.csv', delimiter=",")
        a = np.loadtxt(filename + '_a.csv', delimiter=",").reshape((W.shape[0], 1))
        b = np.loadtxt(filename + '_b.csv', delimiter=",").reshape((W.shape[1], 1))
        return {"W": W, "a": a, "b": b}
