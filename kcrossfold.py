from datasets import Dataset
from sklearn.model_selection import KFold
from neuralREG_tsne import neuralREG_tSNE
import numpy as np
from utils import *
from kNN import kNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# initialization dataset/model
seed = 0
dataset = Dataset(seed)
model_type = 'par' # par/reg/auto/PCA/kernel
data_name = 'MNIST' # MNIST/ COIL20
n_train = 10000
n_test = 10000
batch_size = 1000
d_components = 2

X, labels, X_train, labels_train, X_test, labels_test = dataset.get_data(name=data_name, n_train=n_train, n_test=n_test)

nsplits = 3
kfold = KFold(n_splits= nsplits)
RBM_file = 'Models/weightsRBM/' + data_name + '/' + data_name + str(n_train) + 'dim' + str(d_components)

layer_dim = [784, 500, 500, 2000, d_components]
lambdas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


accuracies = np.zeros(len(lambdas))
losses = []
for i, labda in enumerate(lambdas):
    print('Now training for lambda = %.2f ' % (labda))
    accuracy = 0
    loss = np.zeros(3)
    ifold = 0
    for train, test in kfold.split(X_train):
        ifold += 1
        print('Fold %d for labda = %.2f' %(ifold, labda))
        model = neuralREG_tSNE(epochs=20, batch_size=batch_size, lr=0.01, labda=labda)
        model.load_RBM(RBM_file, layer_dim)
        xtrain = X_train[train]
        xtest = X_train[test]
        # losses
        all_losses = model.train(xtrain)
        loss += all_losses[-1]

        # accuracy
        y_train = model.predict(xtrain)
        nn_classifier = kNN(y_train, labels_train[train])
        y_test = model.predict(X_test[test])
        preds_nn = nn_classifier.predict(y_test)
        acc = np.mean(preds_nn == labels_test[test])
        accuracy += acc
    print('Finished 3-fold for labda = %.2f' % labda)
    print('Accuracy: ' + str(accuracy/nsplits))
    print('Losses: ' + str(loss/nsplits))
    accuracies[i] = accuracy/nsplits
    losses.append(loss/nsplits)
print('')
print(accuracies)
print(losses)

file_path = 'Models/results/' + data_name + str(n_train) + model_type + 'dim' + str(d_components) + 'results.txt'
make_dir(file_path)
with open(file_path, 'w') as file:
    file.write('Results for %d-fold validation on %s %d with model: %s ' %(nsplits, data_name, n_train, model_type))
    file.write('lambdas: ' + str(lambdas))
    file.write('\n')
    file.write('accuracies: ' + str(accuracy/nsplits))
    file.write('\n')
    file.write('t-SNE cost: ')
    for loss in losses:
        file.write(str(loss[1]) + ' ')
    file.write('\n')
    file.write('MSE cost: ')
    for loss in losses:
        file.write(str(loss[2]) + ' ')






