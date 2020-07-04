'''
    Adapted from the MatLab code by Ruslan Salakhutdinov and Geoff Hinton
    Available at: http://science.sciencemag.org/content/suppl/2006/08/04/313.5786.504.DC1
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from RBM import *

learning_rate = 0.001


class RBM_with_linear_hidden_units(RBM):
    '''
    A class that trains a RBM with linear hidden units
    drawn from a unit variance Gaussian whose mean is determined by the input from
    the logistic visible units" (Hinton, 2006)

    The only difference from RBM is how h_probs are generated and h_states are
    sampled.
    '''
    def h_probs(self, v):
        '''
            h_probs is defined differently than in the RBM
            with binary hidden units.

            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert (v.shape[0] == self.v_dim)
        return self.b + np.dot(self.W.T, v)

    def train(self, x, epochs=10, batch_size=100, learning_rate=learning_rate, plot=False, initialize_weights=True):
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
            # visible bias a_i is initialized to ln(p_i/(1-p_i)), p_i = (proportion of examples where x_i = 1)
            # self.a = (np.log(np.mean(x,axis = 1,keepdims=True)+1e-10) - np.log(1-np.mean(x,axis = 1,keepdims=True)+1e-10))
            self.a = np.zeros((self.v_dim, 1))
            self.b = np.zeros((self.h_dim, 1))

        for i in range(epochs):
            print("Epoch %i" % (i + 1))
            np.random.shuffle(x.T)

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
                h_pos_states = h_pos_probs + np.random.normal(0., 1., size=h_pos_probs.shape)  # this line changes, sigma=1

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

            print("Epoch: %d Reconstruction MSE: %.4f elapsed time: %.2f" % (i + 1, error_sum, time() - t0))
            error_sum = 0.


        return self.h_probs(x)

