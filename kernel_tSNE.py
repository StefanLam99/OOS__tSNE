import numpy as np
from time import time
from utils import norm_gauss_kernel,determine_sigma, cond_probs, pca, joint_Q, joint_average_P, profile, plot, distance_matrix, distance_matrix_squared
from datasets import Dataset
from tSNE import *

class kernel_tSNE:
    def __init__(self, d_components=2, initial_dims=30, initialization='PCA', perplexity=40, dof=1., early_exaggeration=4,
                 X_train = None, random_state = 0, y_train = None,c=0.5):

        self.d_components = d_components # dimension of projection space
        self.initial_dims = initial_dims # initial dimensions of data, before applying t-sne
        self.initialization = initialization # initialization method if there is one, defaults to PCA and uses initial_dims
        self.X_train = X_train
        self.y_train = y_train
        self.c = c
        D = distance_matrix(self.X_train)
        self.sigma = determine_sigma(D, self.c)
        self.random_state = random_state


    @profile
    def train(self):
        '''
        train by implementing t sne to obtain y_train
        '''
        if (self.X_train == None):
            print('load data first')
            return
        tSNE = tsne(d_components= self.d_components, initial_dims=30, initialization='PCA', perplexity=40, dof=1.,
                 random_state=self.random_state, grad_method = 'ADAM', max_iter =1000, learning_rate = 0.05)
        self.y_train, cost = tSNE.transform(self.X_train)
        return self.y_train, cost


    def predict(self, X):
        '''
        predicts using kernel t-sne
        '''
        ''' 
        if self.y_train == None:
            print('load the y train data first!')
            return
        elif self.X_train == None:
            print('load the X train data first!')
            return
        '''
        X_test = X
        X_train = self.X_train
        if self.initialization == 'PCA':
            (n, _) = self.X_train.shape
            new_X, _ = pca(np.concatenate((self.X_train, X), axis=0), 30)
            X_train = new_X[0:n,:]
            X_test = new_X[n:,:]

        K_train = norm_gauss_kernel(X_train, X_train, self.sigma)
        K_inverse = np.linalg.pinv(K_train)
        A = np.dot(K_inverse, self.y_train)
        K_test = norm_gauss_kernel(X_train, X_test, self.sigma)
        Y = np.dot(K_test, A)

        return Y
    def load(self, file_path):
        '''
        Load y_train from file_path
        '''
        self.y_train = np.genfromtxt(file_path, delimiter=',')

    def load_data(self, X_train):
        self.X_train = X_train



if __name__ == '__main__':
    '''
    begin = time()
    seed = 0
    dataset = Dataset(seed)
    #X, y, X_train, y_train, X_test, y_test = dataset.get_IRIS_data(n_train=100)
    #X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=6000, n_test=4000)
    #alpha = np.genfromtxt('kernelCoil20alpha.csv', delimiter=',')
    model = kernel_tSNE(random_state=seed, max_iter=100, X_train=X_train, initialization=None)
    #model.load('kernelMNIST10000alpha.csv')
    Y, alpha, cost = model.train()
    np.savetxt('kernelMNIST10000Y.csv', Y, delimiter=',')
    np.savetxt('kernelMNIST10000alpha.csv', alpha, delimiter=',')
    np.savetxt('kernelMNIST10000cost.csv', cost, delimiter=',')
    #np.savetxt('kernelCoil20alpha.csv', alpha, delimiter=',')
    Y = model.predict(X_train)
    #np.savetxt('test.csv' , y_test, delimiter=',')
    #np.savetxt('kerneltrainY.csv', Y, delimiter=',')
    end = time()
    plot(Y, y_train, cmap='Paired',title='Kernel t-SNE: mnist10000 train, ' )
    Y = model.predict(X_test)
    plot(Y, y_test, cmap='Paired', title='Kernel t-SNE: mnist10000 test, ' )
    
    t0 = time()
    seed = 0
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=6000, n_test=4000)
    Y = np.genfromtxt('results/MNIST/ADAMY2.csv', delimiter=',')
    #X, _ = pca(np.concatenate((X_train,X_test), axis=0), 30)
    X_train = X[0:6000,:]
    X_test = X[6000:,:]
    #cond_P, sigma = cond_probs(X_train)
    #np.savetxt('P.csv', cond_P, delimiter=',')
    #cond_P
    #print(sigma)
    D = distance_matrix(X)
    print(D.shape)
    sigma, sigma_first = determine_sigma(D)
    print(sigma.shape)

    print('starting kernel calculatipon')
    K_train = gauss_kernel(X, X, sigma_first)
    K_train = K_train/np.sum(K_train,1).reshape(6000,-1)
    print('elapsed time: %.2f'%(time() - t0))
    print('starting pseudo inverse')
    K_inverse = np.linalg.pinv(K_train)
    #K_inverse = np.linalg.pinv(X_train)

    print(X_train.shape)
    print(K_inverse)
    #print(A)
    print(K_inverse.shape)
    print('elapsed time: %.2f' % (time() - t0))
    A = np.dot(K_inverse, Y)

    Y_train = np.dot(K_train, A)
    #Y_train = np.dot(X_train, A)
    plot(Y_train, y_train, cmap='Paired', title='Kernel t-SNE SIMPLE witb pca sigma first 0.5: mnist10000 train, ')
    A = np.dot(np.dot(np.linalg.inv(np.dot(K_train.T, K_train)), (K_train.T)), Y)

    K_test = gauss_kernel(X_train, X_test, sigma_first)
    K_test = K_test/ np.sum(K_test, 1).reshape(4000, -1)
    Y_test = np.dot(K_test, A)
    #Y_test = np.dot(X_test, A)
    plot(Y_test, y_test, cmap='Paired', title='Kernel t-SNE SIMPLE with pca sigma first 0.5: mnist10000 test, ')

    '''














