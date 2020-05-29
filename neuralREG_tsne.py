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
from keras.losses import mse, binary_crossentropy
import keras.losses
from utils import *
from autoencoder import Autoencoder
class neuralREG_tSNE:
    def __init__(self, data_name ='',d_components=2,  perplexity=40., epochs=100, lr=0.001, random_state=0, batch_size=100,encoder=None, decoder=None,
                 model=None, labda=0.99):
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
        #keras.losses.kl_loss = self.kl_loss


    def build(self, n_input, layer_sizes = np.array([500, 500, 2000]), activations= np.array(['sigmoid','sigmoid','sigmoid'])):
        ''''
        builds the structure for the regularized parametric t-sne network
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
        autoencoder = Model(input, output=[encoded,decoded])
        self.encoder = Model(input, encoded)
        encoded_input = Input(shape=(layer_sizes[0],))
        decoder_layer = autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(loss={'encoded': self.kl_loss, 'decoded': self.mse_loss}, optimizer=Adam(self.lr),
                            loss_weights= [self.labda, 1 - self.labda])

        self.model = autoencoder


    def train(self, X_train):
        """
        Train the regularized parametric t-SNE network
        """
        print('Start training the neural network...')

        begin = time()
        losses = []

        n_sample, n_feature = X_train.shape
        nBatches = int(ceil(n_sample/self.batch_size))
        for epoch in range(self.epochs):
            new_indices = np.random.permutation(n_sample) # shuffle data for new random batches
            X = X_train[new_indices]
            loss = np.zeros(3)

            for i in range(nBatches):

                batch = X[i*self.batch_size:(i+1)*self.batch_size]
                if int(self.labda) > 0:
                    blockPrint()
                    cond_p, _ = cond_probs(batch.copy(), perplexity=self.perplexity)
                    P = joint_average_P(cond_p)
                    enablePrint()
                    all_losses = self.model.train_on_batch(x=batch, y={'encoded': P, 'decoded': batch})
                else: # runs faster
                    all_losses = self.model.train_on_batch(x=batch, y={'encoded': np.zeros((500,500)), 'decoded': batch})
                loss += np.array(all_losses)

            losses.append(loss/nBatches)
            #losses[epoch] = loss/nBatches
            print('Epoch: %.d losses: %.3f| %.3f| %.3f elapsed time: %.2f' % (
            epoch + 1, losses[epoch][0],losses[epoch][1], losses[epoch][2],time() - begin))

        return losses
    def predict(self, X):
        """
        Makes prediction for a given data set X nxD with the autoencoder
        """
        if self.model == None:
            print("Train the model first!")
            return
        return self.model.predict(X)
    def predict_encoder(self, X):
        if self.encoder == None:
            print("Train the model first!")
            return
        return self.encoder.predict(X)
    def predict_decoder(self, X):
        if self.encoder == None:
            print("Train the model first!")
            return
        return self.decoder.predict(X)

    # loading functions:
    def load_model(self, file_path):
        self.model = load_model(file_path)
    def load_encoder(self, file_path):
        self.encoder = load_model(file_path)
    def load_decoder(self, file_path):
        self.decoder = load_model(file_path)
    #losses
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

    def save(self, dir=None):
        if dir==None:
            print("Directory not specified!")
            return
        self.model.save(dir+'/autoencoder'+self.data_name)
        self.encoder.save(dir+'/encoder'+self.data_name)
        self.decoder.save(dir+'/decoder'+self.data_name)


if __name__ == '__main__':
    n_sample = 10000
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train=X_train[0:n_sample,:]
    y_train=y_train[0:n_sample]
    X_test=X_test[0:5000,:]
    y_test=y_test[0:5000]
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    print(K.image_data_format())
    model = neuralREG_tSNE( data_name='MNIST', epochs=1000, batch_size=500, lr=0.001, labda=0.0)
    #model.build_cnn()
    #model.build_nn(n_input=784, activations=np.array(['sigmoid', 'sigmoid', 'relu']))
    ''' 
    model.build(n_input=784)
    model.model.summary()
    model.train(X_train)
    model.save('Models/NN/regNN')
    model.model.summary()
    '''
    model.load_encoder('Models/NN/regNN/encoderMNIST')
    y = model.predict_encoder(X_train)
    plot(y, y_train, title='labda=0.0')
    print(y.shape)
    print(y)

    y = model.predict_encoder(X_test)
    plot(y, y_test, title='labda=0.0')
    print(np.sum(X_test,axis=1))
    print(np.sum(model.predict(X_test),axis=1))