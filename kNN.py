import numpy as np
from utils import timer
from sklearn.metrics import pairwise_distances
class kNN():
    def __init__(self, X, y):
        self.data = X
        self.targets = y

    @timer
    def euclidean_distance(self, X):
        """
        Computes the euclidean distance between the training data and
        a new input example or matrix of input examples X
        """
        # input: single data point
        if X.ndim == 1:
            D = np.sqrt(np.sum((self.data - X)**2, axis=1))

        else:
            n_samples, _ = X.shape

            #D = [np.sqrt(np.sum((self.data - X[i])**2, axis=1)) for i in range(n_samples)]
            D = pairwise_distances(self.data, X) # n x m

        return np.array(D.T)

    @timer
    def predict(self, X, k=1):
        """
        Predicts the classification for an input example or matrix of input examples X
        """
        #  compute distance between input and training data
        dists = self.euclidean_distance(X)

        #  find the k nearest neighbors and their classifications
        if X.ndim == 1:
            if k == 1:
                nn = np.argmin(dists)
                return self.targets[nn]
            else:
                knn = np.argsort(dists)[:k]
                y_knn = self.targets[knn]
                max_vote = max(y_knn, key=list(y_knn).count)
                return max_vote[0]
        else:
            knn = np.argsort(dists)[:, :k]
            y_knn = self.targets[knn]
            if k == 1:
                return y_knn.T[0]
            else:
                n_samples, _ = X.shape
                max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
                return max_votes[0]