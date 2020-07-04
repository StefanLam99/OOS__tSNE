'''
Some utility function used for our techniques.
'''
import sys, os, io, pstats
import numpy as np
import cProfile
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.ticker import NullFormatter
import timeit
from functools import wraps


def timer(func):
    '''
    Wrapper funtion to time functions
    '''
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"The method '{func.__name__}' took: {end_/1000 if end_/1000 > 0 else 0} seconds")
    return _time_it


# Disable
def blockPrint():
    '''
    disables prints
    '''
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    '''
    enables prints
    '''
    sys.stdout = sys.__stdout__


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner


def shannon_entropy(Di, sigma=1.0):
    """
    Compute shannon entropy Hi and corresponding Pi-row of the i'th object
    Note: sigma can be seen as squared in this method
    """
    # print('distance: ' + str(Di))
    Pj = np.exp(-Di.copy() / (2 * sigma))  # the j elements of the "i'th" row
    # print(Pj)
    sumP = sum(Pj)
    Pi = Pj / sumP

    Hj = np.log(Pi) * Pi
    # print(Hj)
    Hi = -np.sum(Hj)
    # print('H: ' + str(H))
    return Hi, Pi


def distance_matrix_squared(X):
    '''
    Computes squared distances of a matrix column wise, according to squared euclidian distance
    '''

    sum_X_squared = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X_squared).T, sum_X_squared)  # distance matrix squared
    return D


def distance_matrix_squared2(X1, X2):
    '''
    Computes squared distances of two different matrices, accordiing to squared eculidian distsance
    X1:  nxD
    X2:  mxD
    returns: mxn
    '''
    (n,_) = X1.shape
    (m,_) = X2.shape
    sum_X1_squared = np.sum(np.square(X1), 1)
    sum_X2_squared = np.sum(np.square(X2), 1)
    D = np.add(np.add(-2 * np.dot(X1, X2.T), sum_X1_squared.reshape(n,-1)).T, sum_X2_squared.reshape(m,-1))  # distance matrix squared
    return D


def distance_matrix(X):
    '''
    Computes distances of a matrix column wise, according to euclidean distance
    '''

    sum_X_squared = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X_squared).T, sum_X_squared)  # distance matrix squared
    D = np.maximum(D,1e-12)
    return np.sqrt(D)


def cond_probs(X=np.array([]), tol=1e-5, perplexity=40):
    """
    Find the conditional probabilities Pj|i such that each gaussian
    has the same perplexity of an NxD matrix X
    """

    # initialize variables
    begin = time()
    (n, d) = X.shape
    P = np.zeros((n, n))  # the conditional probability matrix
    sigma = np.ones((n, 1))
    logU = np.log(perplexity)
    D = distance_matrix_squared(X)

    print("start binary search...")
    t0 = time()
    # Compute the conditonal probabilities
    for i in range(n):

        # Print progress
        if i % 100 == 0:
            print("Computing probablities for point %d of %d took %d seconds" % (i, n, time() - t0))

        # Compute the Gaussian kernel and entropy for the current precision
        sigma_min = -np.inf
        sigma_max = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, Pi) = shannon_entropy(Di, sigma[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # note: H is monotonically increasing in sigma
            if Hdiff < 0:  # increase sigma
                sigma_min = sigma[i].copy()
                if sigma_max == np.inf or sigma_max == -np.inf:  # if we dont have a value for sima_max yet
                    sigma[i] = sigma[i] * 2.
                else:
                    sigma[i] = (sigma[i] + sigma_max) / 2.
            else:  # decrease sigma
                sigma_max = sigma[i].copy()
                if sigma_min == np.inf or sigma_min == -np.inf:
                    sigma[i] = sigma[i] / 2.
                else:
                    sigma[i] = (sigma[i] + sigma_min) / 2.

            # Recompute the values
            (H, Pi) = shannon_entropy(Di, sigma[i])
            Hdiff = H - logU
            # print(Hdiff)
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = Pi
    print("binary search took %.2f seconds" % (time() - t0))
    print("Finding all conditional probabilities took %.2f seconds" % (time() - begin))
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(sigma)))
    return P, np.sqrt(sigma) #note: sigma is squared version


def joint_average_P(cond_P):
    """
    Compute the joint probability matrix according to Pij = Pji =(Pi|j + Pj|i)/2
    """
    (n, d) = cond_P.shape
    P = (cond_P + np.transpose(cond_P)) / (2 * n)
    P = np.maximum(P, 1e-12)
    # P = P/np.sum(P) #normalize the probabilities
    # print('sum of P is ' + str(np.sum(P)))

    return P


def joint_Q(Y, dof=1.):
    """
    Compute the joint probabilities qij of the low dimensional data Y according to the student-t with dof 1
    """
    (n, d) = Y.shape


    sum_Y_squared = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = ((dof+1.)/2) / (1. + (np.add(np.add(num, sum_Y_squared).T, sum_Y_squared))/dof)  # numerator of qij: (1 + ||yi -yj||^2)^-1
    numerator = num
    num[range(n), range(n)] = 0.

    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    return Q, numerator


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions
    """
    print("Preprocessing the data using PCA...")
    t0 = time()
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X)) # eigenvalues, and ``eigenmatrix''
    Y = np.dot(X, M[:, 0:no_dims]).real
    print("Reducing dimension of data with PCA took: %8.2f seconds" % (time() - t0))
    return Y, M


def norm_gauss_kernel(X1, X2, sigma):
    '''
    Calculates the gaussian kernel function for each
    row of X respect to x. sigma is calculated via
    the binary search and perplexity.
    (m, d) = X2.shape
    (n, D) = X1.shape
    returns: mxn
    '''
    D = distance_matrix_squared2(X1,X2)
    (m, n) = D.shape
    print(D.shape)
    K = np.zeros(D.shape)
    for l in range(D.shape[1]):
        K[:,l] = -D[:,l]/(2*(sigma[l]**2))
    kernel = np.exp(K)
    return kernel/np.sum(kernel,1).reshape(m,-1) # reshape it in the size of the X2 the "test" set


def gauss_kernel2(x, X, sigma):
    '''
    Calculates the gaussian kernel function for each
    row of X respect to x. sigma is calculated via
    the binary search and perplexity.
    (m, d) = x.shape
    (n, D) = X.shape
    '''
    (n, D) = X.shape

    k = np.zeros(n)

    for i in range(n):
        k[i] = np.exp(-(np.linalg.norm(x - X[i,:])**2)/(2.*sigma[i]))
    return k


def determine_sigma(D, c):
    '''
    Calculates the sigma as the mean distance to its fifth neighbor for kernel
    t-SNE.
    :param D: distance matrix
    '''
    sorted = np.sort(D,1)
    five_neighbors = sorted[:,1:5]
    sigma = sorted[:,6]
    #sigma = np.mean(five_neighbors,axis=1)
    sigma_first = sorted[:,1]*c # first column are zeros for n*n
    return sigma_first


def rank_matrix(X):
    '''
    Computes the rank matrix for the pairwise distances of X, rows are in ascending order, according to paiwrise distances
    '''
    (n, m) = X.shape
    D = distance_matrix(X)
    sorted = D.argsort()
    ranks = np.zeros(D.shape)
    indices = np.r_[0:n]
    for i in range(n):
        ranks[i,sorted[i,:]] = indices

    return ranks.astype(np.int), D


@timer
def trustworthniness(X, y, k_neighbors):
    '''
    Calculates the trustworthiness of a DR technique, it can be
    seen as the "precision".
    :param X: high dimensional space
    :param y: low dimensional space
    :param n_neighbors: list of numbers, for which you want to calculate the trustworthiness
    '''
    (n, m) = X.shape

    rank_X, dist_X = rank_matrix(X)
    dist_y = distance_matrix(y)

    dist_X = np.argsort(dist_X)
    dist_y = np.argsort(dist_y)

    trusts = np.zeros(len(k_neighbors))
    for l, k in enumerate(k_neighbors):
        X_neighbors = dist_X[:,1:k+1] # we don't count (i,i) as a neighbor
        y_neighbors = dist_y[:,1:k+1]

        # computing the set U of elements in projection space but not in input space
        U = []
        for i in range(n):
            neighbors = []
            for neighbor in y_neighbors[i,:]: # add the neighbors of y which are not neighbors of X to the set U
                if neighbor not in X_neighbors[i,:]:
                    neighbors.append(neighbor)

            U.append(neighbors)

        # computing trustworthiness for the given k_neighbors
        sum = 0
        for i in range(n):
            for j in U[i]:

                sum = sum + (rank_X[i,j] - k)

        if k < (n/2.0):
            scaler = 2/((n*k)*(2*n - 3*k - 1))
        else:
            scaler =  2/(n * (n - k) * (n - k - 1)) # this won't happen probably, in my case but it's the correct formula

        trusts[l] = 1 - sum * scaler

    return trusts


@timer
def continuity(X, y, k_neighbors):
    '''
    Calculates the continuity of a DR technique, it can be
    seen as the "recall".
    :param X: high dimensional space
    :param y: low dimensional space
    :param n_neighbors: list of numbers, for which you want to calculate the trustworthiness
    '''
    (n, m) = X.shape

    dist_X = distance_matrix(X)
    rank_y, dist_y = rank_matrix(y)

    dist_X = np.argsort(dist_X)
    dist_y = np.argsort(dist_y)

    continuities = np.zeros(len(k_neighbors))
    for l, k in enumerate(k_neighbors):
        X_neighbors = dist_X[:,1:k+1] # we don't count (i,i) as a neighbor
        y_neighbors = dist_y[:,1:k+1]

        # computing the set U of elements in projection space but not in input space
        V = []
        for i in range(n):
            neighbors = []
            for neighbor in X_neighbors[i,:]: # add the neighbors of X which are not neighbors of y to the set V
                if neighbor not in y_neighbors[i,:]:
                    neighbors.append(neighbor)

            V.append(neighbors)

        # computing continuities for the given k_neighbors
        sum = 0
        for i in range(n):
            for j in V[i]:
                sum = sum + (rank_y[i,j] - k)

        if k < (n/2.0):
            scaler = 2/((n*k)*(2*n - 3*k - 1))
        else:
            scaler =  2/(n * (n - k) * (n - k - 1))

        continuities[l] = 1 - sum * scaler

    return continuities


def plot(Y, labels, title='',marker = None ,label = False, cmap= None, s=15, save_path=None, linewidth=1, axis = 'off'):
    '''
    "Simple" plotting method
    '''
    fig, ax = plt.subplots()

    if marker == None:
        scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap = cmap, s=s)
    else:
        cmap = cm.get_cmap(cmap, len(marker))
        for i, e in enumerate(marker):
            ax.scatter(Y[:,0][labels==i], Y[:,1][labels==i], c =np.array([cmap(i)]), s=s, linewidths= linewidth, marker = '$'+marker[i]+'$' )

    ax.set_title(title)
    if axis == 'off':
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        fig.patch.set_visible(False)
        ax.axis('off')
    if label:
        plt.legend(*scatter.legend_elements(), loc="upper right", title='Labels', prop={'size': 6}, fancybox=True)
    fig.tight_layout()
    if save_path != None:
        make_dir(save_path)
        plt.savefig(save_path)
    plt.show()


def make_dir(file_path):
    '''
    Makes a directory for a file path if it does not already exist
    '''
    split = file_path.rsplit('/',1)
    dir = split[0]
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

