
from keras.datasets import mnist
from keras import backend as K
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import Adam
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
seed = 0

#file_path = 'results/'+data_name + '/' + data_name + 'costplotALL'
data_name = 'MNIST' # MNIST/ COIL20
d_components = 10
if data_name == 'COIL20':
    n_train = 960
    n_test = 480
    batch_size = 480
elif data_name == 'MNIST':
    n_train = 10000
    n_test = 5000
    batch_size = 1000
dataset = Dataset(seed)
#X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
Csgd = np.genfromtxt('Models/tSNE/'+data_name+ str(n_train)+ 'dim'+str(d_components)+'SGDcost.csv', delimiter=',')
Cadam = np.genfromtxt('Models/tSNE/'+data_name+ str(n_train)+ 'dim'+str(d_components)+'ADAMcost.csv', delimiter=',')
Cgains = np.genfromtxt('Models/tSNE/'+data_name+ str(n_train)+ 'dim'+str(d_components)+'gainscost.csv', delimiter=',')
start = 101
end = 1000
x = range(start,end)
plt.plot(x, Csgd[start:end], c='r')
plt.plot(x, Cgains[start:end], c='green')
plt.plot(x, Cadam[start:end], c='b')

plt.xticks(np.arange(0,1001,100))
plt.xlim(0,1000)

plt.xlabel('Iteration')
plt.ylabel('t-SNE cost')
plt.title('%d-dimensional projection' % d_components, fontsize=16)
plt.legend(['SGD','a-SGD with momentum','Adam' ], loc='upper right',frameon=True, fancybox=True)
plt.savefig('Models/plots/' + data_name + str(n_train) + 'dim' + str(d_components))
plt.show()


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
        v_states = np.random.binomial(1, v_probs)
        h_probs = self.h_probs(v_states)
        h_states = h_probs + np.random.normal(0., 1., size=h_probs.shape)  # this line changes
    return v_states, h_states


# visisble:

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