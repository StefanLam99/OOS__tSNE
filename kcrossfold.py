from datasets import Dataset
from sklearn.model_selection import KFold
from neuralREG_tsne import neuralREG_tSNE
import numpy as np
from utils import *
from kNN import kNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 0
nsplits = 3
kfold = KFold(n_splits= nsplits)
RBM_file = 'Models/weightsRBM/MNIST60000'
d_components =2
layer_dim = [784, 500, 500, 2000, d_components]
lambdas = [0.5, 0.7, 0.9, 0.99]


dataset = Dataset(seed)
X, labels, X_train, labels_train, X_test, labels_test = dataset.get_MNIST_data(n_train=3000, n_test=10000)

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
        model = neuralREG_tSNE(epochs=5, batch_size=1000, lr=0.05, labda=labda)
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






