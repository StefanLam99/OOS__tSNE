# RBM class with gaussian visible units
# added this class for real-valued input data
import numpy as np
import random
import matplotlib.pyplot as plt

from RBM import *

learning_rate = 0.001


class RBM_with_linear_visible_units(RBM):

    def v_probs(self, h):
        '''
            v_probs is defined differently than in the RBM
            with binary hidden units.

            Input:
            - h has shape (h_dim,m)
            - a has shape (v_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert (h.shape[0] == self.h_dim)
        return self.a + np.dot(self.W, h)

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
                # get the next batch, this part changes for linear visible units
                v_pos_probs = x[:, j * batch_size:(j + 1) * batch_size] # real valued
                v_pos_states = v_pos_probs + np.random.normal(0., 1., size=v_pos_probs.shape) # sigma = 1

                # get hidden probs, positive product, and sample hidden states
                h_pos_probs = self.h_probs(v_pos_states)
                pos_prods = v_pos_states[:, np.newaxis, :] * h_pos_probs[np.newaxis, :, :]
                h_pos_states = np.random.binomial(1, h_pos_probs)  # gibbs sampling step

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

            print("Reconstruction MSE = %.2f" % error_sum)
            error_sum = 0.


        return

    def gibbs_sampling(self, n=1, m=1, v=None):
        '''
            n - number of iterations of blocked Gibbs sampling
        '''
        if v is None:
            v_probs = np.full((self.v_dim, m), 0.5)
            v = np.random.binomial(1, v_probs)

        h_probs = self.h_probs(v)
        h_states = np.random.binomial(1, h_probs)
        for i in range(n):
            v_probs = self.v_probs(h_states)
            v_states = v_probs + np.random.normal(0., 1., size=v_probs.shape)# this line changes
            h_probs = self.h_probs(v_states)
            h_states = np.random.binomial(1, h_probs)
        return v_states, h_states