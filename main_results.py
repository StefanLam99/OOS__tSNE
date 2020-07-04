'''
Main to obtain the performance measures of our techniques
'''

from utils import *
from kernel_tSNE import *
from pca_tSNE import *
from reg_tSNE import *
from par_tSNE import *
from datasets import Dataset
import numpy as np
from kNN import kNN
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.markers as mark
from time import time

seed = 0
dataset = Dataset(seed)
model_type = 'par' # par/reg/auto/PCA/kernel
model_types = [ 'PCA', 'kernel','auto', 'par', 'reg']
data_name = 'COIL20' # MNIST/ COIL20
if data_name == 'COIL20':
    n_train = 960
    n_test = 480
elif data_name == 'MNIST':
    n_train = 10000
    n_test = 5000
n_neigbors = 200 # range of neighbors for continuity/trustworthiness
d_componentss = [2, 10, 20] # should be same range as the thetas
thetas = [0.99, 0.9, 0.7 ] # MNIST:[0.9,0.1,0.01], COIL20:[0.99, 0.9, 0.7 ]
save_plot = True
plot_trusts_conts = True
evaluate = False
X, labels, X_train, labels_train, X_test, labels_test = dataset.get_data(data_name,n_train, n_test )
c = 0.05 # small factor for kernel t-SNE


# trusts and continiuties
all_trusts = np.zeros(shape=(n_neigbors, 5))
all_conts = np.zeros(shape=(n_neigbors, 5))
k_neighbors = list(range(1,n_neigbors+1))

for i, model_type in enumerate(model_types):
    for d_components, theta in zip(d_componentss, thetas):
        print('loading ' + str(model_type) + ' model')
        if model_type == 'PCA':
            model = PCA_tSNE(initial_dim=d_components)
            model.train(X_test) # did not make a load function for pca
        elif model_type == 'kernel':
            model = kernel_tSNE(d_components=d_components, X_train=X_train, c=c, initialization= None)
            model.load('Models/tSNE/'+data_name+ str(n_train) + 'dim' + str(d_components) + 'ADAM'+'Y2.csv')
            model.train() # did not save the alphas...
        elif model_type == 'par':
            model = neural_tSNE(d_components=d_components)
            model.load_model('Models/parametric/'+ data_name + str(n_train) + model_type + 'dim' + str(d_components))
        elif model_type == 'auto':
            model = neuralREG_tSNE(d_components=d_components)
            model.load_model('Models/autoencoder/' + data_name + str(n_train) + model_type + 'dim' + str(d_components))
        elif model_type == 'reg':
            model = neuralREG_tSNE(d_components=d_components)
            model.load_model('Models/regularized/' + data_name + str(n_train) + model_type + 'dim' + str(d_components) + '_' +str(theta))


        if model_type != 'reg':
            theta = -1

        # predictions on train set
        y_train = model.predict(X_train)

        # predictions on test set
        y_test = model.predict(X_test)

        if evaluate:
            nn_classifier = kNN(y_train, labels_train) #1-nn classifier
            preds_nn = nn_classifier.predict(y_test)

            #evaluating performance
            print('evaluating performance')
            trusts = trustworthniness(X_test, y_test, k_neighbors)
            continuities = continuity(X_test, y_test, k_neighbors)
            np.savetxt('Models/results/trusts/' + data_name + str(n_train) +model_type+ 'dim' + str(d_components)+'trusts.csv', trusts, delimiter=',')
            np.savetxt('Models/results/conts/' + data_name + str(n_train) +model_type+ 'dim' + str(d_components)+'conts.csv', continuities,delimiter=',')  # plotting trusts and continuities
            all_trusts[:,i] = trusts
            all_conts[:,i] = continuities
            print('Performance for ' +model_type+' model dim is ' +str(d_components)+' with theta = ' + str(theta))
            print('k_neighbors: ' + str(k_neighbors))
            print('Trusts: ' + str(trusts))
            print('Continuities: ' + str(continuities))
            acc = preds_nn == labels_test
            print('Accuracy: %.4f' % np.mean(acc))
            print('error: %.4f' % (np.array([1])-np.mean(acc)))
            print('')

        if save_plot:
            train_plot_path = 'Models/plots/' + data_name + str(n_train)+ model_type + 'dim' + str(d_components) + 'trainScatter'
            test_plot_path = 'Models/plots/' + data_name + str(n_train) +model_type+ 'dim' + str(d_components) + 'testScatter'
        if data_name == 'MNIST' and d_components == 2:
            plot(Y=y_train, labels=labels_train, s=1, linewidth=0.2, cmap='Paired',save_path=train_plot_path)
            plot(Y=y_test, labels=labels_test, s=1, linewidth=0.2, cmap='Paired', save_path=test_plot_path)
        elif data_name == 'COIL20' and d_components == 2:
            plot(Y=y_train, labels=labels_train, s=1, linewidth=0.3, cmap='nipy_spectral',save_path=train_plot_path)
            plot(Y=y_test, labels=labels_test, s=1, linewidth=0.3, cmap='nipy_spectral',save_path=test_plot_path)

# plot the trustworthiness-continuity curves
if plot_trusts_conts:
    for d_components in [2, 10, 20]:
        for model_type in model_types:
            trusts_path = 'Models/results/trusts/' + data_name + str(n_train) + model_type + 'dim' + str(
                d_components) + 'trusts.csv'
            conts_path = 'Models/results/conts/' + data_name + str(n_train) + model_type + 'dim' + str(
                d_components) + 'conts.csv'

            trusts = np.genfromtxt(trusts_path, delimiter=',')
            conts = np.genfromtxt(conts_path, delimiter=',')
            if model_type == 'par':

                plt.plot(conts, trusts, '-', marker=mark.CARETRIGHT, markevery=50, alpha=0.5)
            else:
                plt.plot(conts, trusts, '-', marker=mark.CARETRIGHT, markevery=50, alpha=0.5)

        plt.xlabel('Continuities', fontsize=14)
        plt.xlim([None, 1.009])
        plt.ylim([None, 1.009])
        plt.ylabel('Trustworthiness', fontsize=14)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title('%d-dimensional projection' % d_components, fontsize=16)
        if d_components == 2:
            plt.legend(['PCA', 'Kernel t-SNE', 'Autoencoder', 'Parametric t-SNE', 'RP t-SNE'], loc='upper left',
                       frameon=True, fancybox=True, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('Models/results/trusts/' + data_name + str(n_train) + 'dim' + str(d_components) + 'trustsconts')
        plt.show()








