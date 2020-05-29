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
    def __init__(self, d_components=2, perplexity=40., epochs=100, lr=0.001, random_state=0, batch_size=100, model=None, labda=0.99):
        self.d_components = d_components
        self.perplexity = perplexity
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.batch_size = batch_size
        self.model = model
        self.labda = labda
        #keras.losses.kl_loss = self.kl_loss

    def build_nn(self, n_input, layer_sizes = np.array([500, 500, 2000]), activations= np.array(['sigmoid','sigmoid','relu'])):
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

    def build_regnn(self, n_input, layer_sizes = np.array([500, 500, 2000]), activations= np.array(['sigmoid','sigmoid','relu'])):
        ''''
        builds the structure for a multilayer feedforward neural network
        '''
        if self.model is not None:
            self.model = None
            print('Deleting current model for new model...')

        input = Input(shape=(n_input,), name='input')
        encoded = Dense(layer_sizes[0], activation=activations[0])(input) # input layer
        for size, activation in zip(layer_sizes[1:], activations[1:]):  # hidden layers for encoder
            encoded = Dense(size, activation=activation)(encoded)
        encoded = Dense(self.d_components, activation='relu', name='encoded')(encoded)
        decoded = Dense(layer_sizes[-1], activation=activations[-1])(encoded) # start of decoder
        for size, activation in zip(np.flip(layer_sizes)[1:], np.flip(activations)[1:]): # hidden layers for decoder
            decoded = Dense(size, activation=activation)(decoded)
        decoded = Dense(n_input, name='decoded')(decoded)

        autoencoder = Model(input, output=[encoded,decoded])
        #encoder = Model(input, encoded)
        #decoder = Model(Input(shape=(self.d_components,)), decoded)

        autoencoder.compile(loss={'encoded': self.kl_loss, 'decoded': self.mse_loss}, optimizer=Adam(self.lr),
                            loss_weights= [self.labda, 1 - self.labda])
        self.model =autoencoder

    def build_cnn(self):
        nb_filters = 32
        nb_pool = 2
        nb_conv = 3
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(1,28,28), data_format='channels_first'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))
        self.model.compile(loss=self.kl_loss, optimizer=Adam(self.lr))

    def train(self, X_train):
        """
        Trains the neural network
        """
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

    def save(self, file_path = None):
        if file_path==None:
            print("No file path specified!")
            return
        self.model.save(file_path)


if __name__ == '__main__':
    n_sample = 4000
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train=X_train[0:n_sample,:]
    y_train=y_train[0:n_sample]
    X_test=X_test[0:1000,:]
    y_test=y_test[0:1000]
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    '''
    autoencoder = Autoencoder(layer_dims=[784, 500, 500, 2000, 2])
    autoencoder.pretrain(X_train.T, epochs=15, num_samples= 1000)
    autoencoder.save("pretrained_weights")
    model = autoencoder.unroll()

    model = neural_tSNE(model=model, lr=0.05)
    model.save('rbmshit')
    model.train(X_train)

    
    model = neural_tSNE(epochs=500, batch_size=200, lr=0.015)
    
    
    losses = model.train(X_train)
    model.save('MNISTdrop5' + str(n_sample))
    np.savetxt('lossesdrop5.csv', losses, delimiter=',')
    '''
    print(K.image_data_format())
    model = neural_tSNE( batch_size=200, lr=0.001)
    #model.build_cnn()
    #model.build_nn(n_input=784, activations=np.array(['sigmoid', 'sigmoid', 'relu']))

    model.build_regnn(n_input=784)
    model.model.summary()
    model.train(X_train)
    #model.save('normalshit')
    model.model.summary()

    y = model.predict(X_train)
    plot(y, y_train, title='normal5')
    print(y.shape)
    print(y)

    y = model.predict(X_test)
    plot(y, y_test, title='normal')
