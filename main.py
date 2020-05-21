import numpy as np
from tSNE import *
from time import time
from sklearn import datasets
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from datasets import Dataset
import cProfile
def mainMNIST():
    begin = time()
    ''' 
    data = np.genfromtxt('data/MNIST/mnist_train.csv', delimiter=',')
    n_sample = 1000
    y = data[0:n_sample, 0]
    X = data[0:n_sample, 1:]
    X.astype(np.float32, copy=False)
    perplexities = [5, 30, 50, 100]
    X = X / 255
    '''
    seed = 0
    grad_method = 'ADAM'
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data()
    X = X_train
    y = y_train
    model = tsne(random_state=20, grad_method='ADAM', perplexity=30, max_iter=1000, data_name='MNIST')
    Y = model.transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='Paired', s=1)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_title('MNIST learning rate is %d sample is %s' % (2, grad_method))
    plt.legend(*scatter.legend_elements(), loc="lower right", title='Labels', prop={'size': 6}, fancybox=True)
    fig.tight_layout()
    plt.show()

def mainIRIS():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    X = normalize(X)

    model = tsne(random_state=20, data_name='IRIS', grad_method='gains', perplexity=30, max_iter=1000)
    Y = model.transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='Paired', s=8)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_title('learning rate is %d sample is ADAM' % (2))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend(*scatter.legend_elements(), loc="lower right", title='Labels', prop={'size': 6}, fancybox=True)
    fig.tight_layout()
    plt.show()

def mainCoil20():
    seed = 0
    grad_method = 'ADAM'
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()

    model = tsne(random_state=seed, data_name='IRIS', grad_method=grad_method, perplexity=30, max_iter=1000)
    Y = model.transform(X)
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap='Paired', s=8)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_title('learning rate is %d sample is %s' % (2, grad_method))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend(*scatter.legend_elements(), loc="lower right", title='Labels', prop={'size': 6}, fancybox=True)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    mainMNIST()
    #mainIRIS()
    #mainCoil20()

