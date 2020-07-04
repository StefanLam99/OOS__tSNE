'''
Class to pretrain the weights of an (auto)encoder network with RBMs
'''
import numpy as np
import pandas as pd
import os.path
from RBM import *
from RBM_linear_hidden import *
from RBM_linear_visible import *
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from utils import profile
learning_rate = 0.01


class Autoencoder:

    def __init__(self, layer_dims):
        '''
            Inputs:
            - layer_dims = A list of the layer sizes, visible first, latent last
            Note that the number of hidden layers in the unrolled autoencoder 
            will be twice the length of layer_dims. 
        '''

        self.latent_dim = layer_dims[-1]
        self.v_dim = layer_dims[0]
        self.num_hidden_layers = len(layer_dims) - 1
        self.layer_dims = layer_dims
        self.W = []
        self.b = []
        self.a = []
        self.pretrained = False
        return

    @classmethod
    def pretrained_from_file(cls, filename):
        '''
            Initialize with pretrained weights from a file.
            Still needs to be unrolled.
        '''
        i = 0
        weights = []
        layer_dims = []

        while os.path.isfile(filename + "_" + str(i) + "_a.csv"):  # load the next layer's weights
            weights.append(RBM.load_weights(filename + "_" + str(i)))  # load the next dict of weights
            layer_dims.append(np.shape(weights[i]['W'])[0])
            i = i + 1
        layer_dims.append(np.shape(weights[i - 1]['W'])[1])

        rbm = cls(layer_dims)

        for i in range(rbm.num_hidden_layers):
            rbm.W.append(weights[i]['W'])
            rbm.a.append(weights[i]['a'])
            rbm.b.append(weights[i]['b'])

        rbm.pretrained = True

        return rbm

    @profile
    def pretrain(self, x, epochs, num_samples=50000):
        '''
            Greedy layer-wise training
            The last layer is a RBM with linear hidden units
            shape(x) = (v_dim, number_of_examples)
        '''
        RBM_layers = []

        # initialize RBMs
        for i in range(self.num_hidden_layers):
            if i == 0:
                RBM_layers.append(RBM_with_linear_visible_units(self.layer_dims[i], self.layer_dims[i + 1])) # linear input
            elif  (i < self.num_hidden_layers -1):
                RBM_layers.append(RBM(self.layer_dims[i], self.layer_dims[i + 1])) #
            else:
                RBM_layers.append(RBM_with_linear_hidden_units(self.layer_dims[i], self.layer_dims[i + 1])) # linear output

        # train stack of RBMs
        for i in range(self.num_hidden_layers):  # train RBM's 
            print("Training RBM layer %i" % (i + 1))
            x = RBM_layers[i].train(x, epochs)  # train the ith RBM


            self.W.append(RBM_layers[i].W)  # save trained weights
            self.b.append(RBM_layers[i].b)
            self.a.append(RBM_layers[i].a)

        self.pretrained = True

        return

    def unroll(self):
        '''
            Unrolls the pretrained RBM network
            and sets hidden layer parameters to pretrained values.
        '''
        if self.pretrained == False:
            print("Model not pretrained.")
            return

        # define keras model structure
        inputs = Input(shape=(self.v_dim,))
        x = inputs

        # build encoder part
        for i in range(self.num_hidden_layers - 1):
            weights = [self.W[i], self.b[i].flatten()]
            x = Dense(self.layer_dims[i + 1],
                      activation='sigmoid',
                      weights=weights)(x)

        weights = [self.W[self.num_hidden_layers-1], self.b[self.num_hidden_layers-1].flatten()]
        encoded = Dense(self.layer_dims[self.num_hidden_layers],
                  weights=weights, activation='linear', name='encoded')(x)
        x = encoded

        # build decoder part
        for i in range(self.num_hidden_layers):
            weights = [self.W[self.num_hidden_layers - i - 1].T, self.a[self.num_hidden_layers - i - 1].flatten()]
            if (i == self.num_hidden_layers -1):
                x = Dense(self.layer_dims[self.num_hidden_layers - i - 1],
                          activation='sigmoid',
                          weights=weights, name='decoded')(x)
            else:
                x = Dense(self.layer_dims[self.num_hidden_layers - i - 1],
                          activation='sigmoid',
                          weights=weights)(x)



        autoencoder = Model(inputs, outputs=[encoded, x])
        encoder = Model(inputs, encoded)
        # make decoder
        decoder_input = Input(shape=(self.layer_dims[-1],))
        decoded = decoder_input

        for i in range(len(self.layer_dims)-1):
            decoded = autoencoder._layers[len(self.layer_dims)+i](decoded)

        decoder = Model(decoder_input, decoded)

        return autoencoder, encoder, decoder

    def save(self, filename):
        '''
        Saving the weights of the pretrained RBM
        '''

        if self.pretrained == True:
            for i in range(self.num_hidden_layers):
                weights = {"W": self.W[i], 'a': self.a[i], 'b': self.b[i]}
                RBM.save_weights(weights, filename + "_" + str(i))
        else:
            print("No pretrained weights to save.")
        return