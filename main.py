import numpy as np
from tSNE import *
from time import time
from sklearn import datasets
from sklearn.preprocessing import normalize, minmax_scale
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from datasets import Dataset
from utils import plot, make_dir
import cProfile
def mainMNIST():
    seed = 0
    grad_method = 'SGD'
    lr = 500
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=6000)
    X = X_train
    y = y_train
    model = tsne(random_state=0, grad_method=grad_method, perplexity=40, max_iter=1000, data_name='MNIST', learning_rate=lr)
    Y = model.transform(X)
    plot(Y, y)
def mainIRIS():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    X = normalize(X)

    model = tsne(random_state=20, data_name='IRIS', grad_method='gains', perplexity=40, max_iter=1000)
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
    grad_method = 'SGD'
    file_path = 'results/COIL20/labels.csv'
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
    #make_dir(file_path)
    #np.savetxt(file_path, y_train, delimiter=',')
    model = tsne(learning_rate=500,random_state=seed, data_name='COIL20', grad_method=grad_method, perplexity=40, max_iter=1000)
    Y = model.transform(X)
    plot(Y, y, cmap='tab20b')

def mainLETTER():
    seed = 0
    grad_method = 'ADAM'
    n_sample = 1000
    dataset = Dataset(seed)
    X, y, X_train, y_train, X_test, y_test = dataset.get_LETTER_data(n_train=n_sample)

    model = tsne(learning_rate=0.5,random_state=seed, data_name='LETTER', grad_method=grad_method, perplexity=40, max_iter=1000)
    Y = model.transform(X_train)
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    cmap = cm.get_cmap('gist_earth', len(alphabet))
    print(X_train)
    for i, letter in enumerate(alphabet):
        plt.scatter(Y[:, 0][y_train == i + 1], Y[:, 1][y_train == i + 1], s=20, c=np.array([cmap(i)]),
                    marker='$' + letter + '$', linewidths=0.2)
    plt.show()
if __name__ == '__main__':
    mainMNIST()
    #mainIRIS()
    #mainCoil20()
    #mainLETTER()

