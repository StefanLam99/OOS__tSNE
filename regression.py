import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tSNE import cond_probs, joint_average_P
from time import time
from tSNE import pca
from datasets import Dataset
import sys
np.set_printoptions(threshold=sys.maxsize)
begin = time()
#DATA:
Y_MA= np.genfromtxt('results/YADAM.csv', delimiter=',')
Y_Mg = np.genfromtxt('results/Ygains.csv', delimiter=',')
Y_IA = np.genfromtxt('results/IRISYADAM.csv', delimiter=',')
Y_Ig = np.genfromtxt('results/IRISYgains.csv', delimiter=',')
(n, d) = Y_MA.shape
dataset = Dataset(0)

data = np.genfromtxt('data/MNIST/mnist_train.csv', delimiter=',')
X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data()
P = np.genfromtxt('results/MNISTProbabilitiesADAM.csv', delimiter =',')
n_sample = 3000
test_sample = 1000
labels = data[0:n_sample, 0]

#X
X = X_train


#X_test
labels_test = y_test

models = []
for i in range(d):
    model = LinearRegression()
    #model.fit(X, Y_MA[:,i])
    model.fit(P, Y_MA[:, i])
    models.append(model)

fig, subplots = plt.subplots(1,2)
subplots = subplots.flatten()

ax1 = subplots[0]
#y1 = models[0].predict(X)
#y2 = models[1].predict(X)
y1 = models[0].predict(P)
y2 = models[1].predict(P)
print(y1)
test_cond = cond_probs(X_test, perplexity=30)
P_test = joint_average_P(test_cond)
ax1.set(aspect='equal')
ax1.set_title('train data')
ax1.set_xlabel('dim_1')
ax1.set_ylabel('dim_2')
scatter = ax1.scatter(y1, y2, c=labels, cmap='Paired', s=2)

ax1 = subplots[1]
#y1 = models[0].predict(X_test)
#y2 = models[1].predict(X_test)
y1 = models[0].predict(P_test)
y2 = models[1].predict(P_test)
ax1.set(aspect='equal')
ax1.set_title('test data')
ax1.set_xlabel('dim_1')
ax1.set_ylabel('dim_2')
scatter = ax1.scatter(y1, y2, c=labels_test, cmap='Paired', s=2)

plt.legend(*scatter.legend_elements(), loc="lower right", title='Labels', prop={'size': 6}, fancybox=True,
           bbox_to_anchor=(-0.2, 0), )
fig.tight_layout()
plt.show()

plt.title('test data')
plt.xlabel('dim_1')
plt.ylabel('dim_2')

scatter2= plt.scatter(y1, y2, c=labels_test, cmap='Paired', s=6)
plt.legend(*scatter2.legend_elements(), loc="lower right", title='Labels', prop={'size': 6}, fancybox=True,
           bbox_to_anchor=(-0.2, 0), )
plt.show()
print(y1)

print('it took: %.2f seconds' % (time() - begin))
print(labels_test)
print(X)
print(X_test)
print(P.shape)