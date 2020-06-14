from matplotlib import pyplot as plt
import numpy as np
import keras
from tqdm import tqdm
from keras import backend as K
from keras.layers import Input, LeakyReLU
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from math import ceil
from RBM import RBM
from keras.losses import mse, binary_crossentropy
import keras.losses
from utils import *
from autoencoder import Autoencoder
class neuralREG_tSNE:
    def __init__(self, data_name ='',d_components=2,  perplexity=40., epochs=100, lr=0.001, random_state=0, batch_size=100,encoder=None, decoder=None,
                 model=None, labda=0.99, toTrain = True):
        self.d_components = d_components
        self.perplexity = perplexity
        self.data_name = data_name
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.batch_size = batch_size
        self.model = model # gonna be the autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.labda = labda
        ''' 
        if toTrain is False:
            #keras.losses.kl_loss = self.kl_loss
            keras.losses.kl_loss = self.mse_loss
            keras.losses.kl_loss = mse
        '''

    def build(self, n_input, layer_sizes = np.array([500, 500, 2000]), activations= np.array(['sigmoid','sigmoid','sigmoid'])):
        ''''
        builds the structure for the regularized parametric t-sne network: autoencoder
        '''
        if self.model is not None:
            self.model = None
            print('Deleting current model for new model...')

        input = Input(shape=(n_input,), name='input')# input layer
        # hidden layers for encoder
        encoded = Dense(layer_sizes[0], activation=activations[0])(input)
        for size, activation in zip(layer_sizes[1:], activations[1:]):
            encoded = Dense(size, activation=activation)(encoded)

        encoded = Dense(self.d_components,activation='linear', name='encoded')(encoded) # low dimensional representation

        decoded = Dense(layer_sizes[-1], activation=activations[-1])(encoded) # start of decoder
        for size, activation in zip(np.flip(layer_sizes)[1:], np.flip(activations)[1:]): # hidden layers for decoder
            decoded = Dense(size, activation=activation)(decoded)
        decoded = Dense(n_input,activation=activations[0], name='decoded')(decoded)

        # define autoencoder,encoder and decoder models
        autoencoder = Model(input, outputs=[encoded,decoded])
        self.encoder = Model(input, encoded)
        encoded_input = Input(shape=(layer_sizes[0],))
        decoder_layer = autoencoder._layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        self.model = autoencoder


        decoder_input = Input(shape=(self.d_components,))
        decoded = decoder_input
        n_encoder_layers = int(len(self.model._layers) / 2) + 1
        for i in range(n_encoder_layers-1):

            decoded = self.model._layers[n_encoder_layers+i](decoded)
        self.decoder = Model(decoder_input, decoded)
        self.set_compiler()
        autoencoder.summary()
        self.encoder.summary()
        self.decoder.summary()

    def train(self, X_train):
        """
        Train the regularized parametric t-SNE network
        """
        print('Start training the neural network...')

        begin = time()
        losses = []

        n_sample, n_feature = X_train.shape
        nBatches = int(n_sample/self.batch_size)
        for epoch in range(self.epochs):
            new_indices = np.random.permutation(n_sample) # shuffle data for new random batches
            X = X_train[new_indices]
            loss = 0

            for i in range(nBatches):

                batch = X[i*self.batch_size:(i+1)*self.batch_size]
                if self.labda > 0: # runs faster this way
                    blockPrint()
                    cond_p, _ = cond_probs(batch.copy(), perplexity=self.perplexity)
                    P = joint_average_P(cond_p)
                    enablePrint()
                    if self.labda == 1: # parametric t-sne
                        all_losses = self.encoder.train_on_batch(x=batch, y={'encoded': P})
                    else: # regularized parametric t-sne
                        all_losses = self.model.train_on_batch(x=batch, y={'encoded': P, 'decoded': batch})
                else: # autoencoder with mse loss
                    all_losses = self.model.train_on_batch(x=batch, y={'decoded': batch})

                loss += np.array(all_losses)


            losses.append(loss/nBatches)
            #losses[epoch] = loss/nBatches
            print('Epoch: %.d elapsed time: %.2f losses: %s ' % (epoch + 1,time() - begin, losses[epoch]))

        return losses

    def predict(self, X):
        """
        Makes encoded prediction for a given data set X nxD with the autoencoder
        """
        if self.model == None:
            print("Train the model first!")
            return
        Y = self.model.predict(X)

        return Y[0]

    def predict_encoder(self, X):
        if self.encoder == None:
            print("Load the encoder first!")
            return
        Y = self.model.predict(X)

        return Y[0]

    def predict_decoder(self, X):
        if self.encoder == None:
            print("Train the decoder first!")
            return
        Y = self.model.predict(X)
        return Y[1]

    # loading functions:
    def load_model(self, file_path):
        '''
        load the autoencoder
        '''
        # setting up the autoencoder
        self.model = load_model(file_path, custom_objects={'kl_loss': self.kl_loss, 'mse_loss': self.mse_loss})
        self.set_compiler()

        # acces the encoder and decoder of the autoencoder:
        n_encoder_layers = int(len(self.model._layers)/2) + 1

        self.encoder = Model(self.model.input, self.model.layers[n_encoder_layers-1].output)
        self.encoder.compile(loss={'encoded': self.kl_loss}, optimizer=Adam(self.lr))
        decoder_input = Input(shape=(self.d_components,))
        decoded = decoder_input
        for i in range(n_encoder_layers-1):

            decoded = self.model._layers[n_encoder_layers+i](decoded)
        self.decoder = Model(decoder_input, decoded)

    def load_RBM(self, file_path, layer_sizes):
        '''
        load the autoencoder via the RBMs
        '''
        RBM = Autoencoder(layer_sizes)
        RBM = RBM.pretrained_from_file(file_path)
        self.model, self.encoder, self.decoder = RBM.unroll()
        self.set_compiler()
        self.encoder.compile(loss={'encoded': self.kl_loss}, optimizer=Adam(self.lr))


    # losses
    def mse_loss(self, X, Y):
        return mse(X,Y)

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

    def set_compiler(self):
        if self.labda == 1:
            self.model.compile(loss={'encoded': self.kl_loss}, optimizer=Adam(self.lr))
        elif self.labda == 0:
            self.model.compile(loss = {'decoded': 'mse'}, optimizer=Adam(self.lr))
        else:
            self.model.compile(loss={'encoded': self.kl_loss, 'decoded': self.mse_loss}, optimizer=Adam(self.lr),
                                loss_weights=[self.labda, 1-self.labda])
    def save(self, file_path=None):
        if dir==None:
            print("File_path not specified!")
            return
        make_dir(file_path)
        self.model.save(file_path)

