from utils import *
from kernel_tSNE import *
from pca_tSNE import *
from neuralREG_tsne import *
from neural_tSNE import *
from datasets import Dataset
import numpy as np
from kNN import kNN
from time import time

seed = 0
dataset = Dataset(seed)
model_type = 'kernel' # par/reg/auto/PCA/kernel
data_name = 'MNIST' # MNIST/ COIL20
if data_name == 'COIL20':
    n_train = 960
    n_test = 480
elif data_name == 'MNIST':
    n_train = 10000
    n_test = 5000
d_componentss = [2,10, 20]
X, y, X_train, labels_train, X_test, labels_test = dataset.get_data(data_name,n_train, n_test )


labdas = [0.9, 0.01, 0.1] #[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
c = 0.01
for d_components, labda in zip(d_componentss, labdas):
    print('loading ' + str(model_type) + ' model')
    if model_type == 'PCA':
        model = PCA_tSNE(initial_dim=d_components)
        model.train(X_train)
    elif model_type == 'kernel':
        model = kernel_tSNE(d_components=d_components, X_train=X_train, c=c, initialization=                        None)
        model.load('Models/tSNE/'+data_name+ str(n_train) + 'dim' + str(d_components) + 'ADAM'+'Y2.csv')
    elif model_type == 'par':
        model = neural_tSNE(d_components=d_components)
        model.load_model('Models/parametric/'+ data_name + str(n_train) + model_type + 'dim' + str(d_components))
    elif model_type == 'auto':
        model = neuralREG_tSNE(d_components=d_components)
        model.load_model('Models/autoencoder/' + data_name + str(n_train) + model_type + 'dim' + str(d_components))
    elif model_type == 'reg':
        model = neuralREG_tSNE(d_components=d_components)
        model.load_model('Models/regularized/' + data_name + str(n_train) + model_type + 'dim' + str(d_components) + '_' +str(labda))


    if model_type != 'reg':
        labda = -1


    # training
    y_train = model.predict(X_train)
    nn_classifier = kNN(y_train, labels_train)

    # testing
    print('predicting test sample')
    y_test = model.predict(X_test)
    preds_nn = nn_classifier.predict(y_test)

    #evaluating performance
    print('evaluating performance')
    k_neighbors = [10,12,30]
    trusts = trustworthniness(X_test, y_test, k_neighbors)
    continuities = continuity(X_test, y_test, k_neighbors)

    print('Performance for ' +model_type+' model dim is ' +str(d_components)+' with lambda = ' + str(labda))
    print('k_neighbors: ' + str(k_neighbors))
    print('Trusts: ' + str(trusts))
    print('Continuities: ' + str(continuities))
    acc = preds_nn == labels_test

    print('Accuracy: %.4f' % np.mean(acc))
    print('error: %.4f' % (np.array([1])-np.mean(acc)))
    print('')
    if d_components == 2:
        plot(Y=y_train, labels=labels_train, cmap='gist_rainbow',title='train: '+ model_type +' '+ data_name + 'labda = ' + str(labda), s=1, linewidth=0.2)
        plot(Y=y_test, labels=labels_test, cmap='gist_rainbow',title='test '+ model_type +' '+ data_name+ 'labda = ' + str(labda), s=1, linewidth=0.2)













