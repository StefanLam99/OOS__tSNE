# Implementation of t-SNE, for reference see (Maaten & Hinton 2009)
# In addition I implement the original t-SNE also with ADAM gradient descent
# author: Stefan Lam
import numpy as np
from time import time
from utils import profile, shannon_entropy, cond_probs, joint_average_P, joint_Q, pca, make_dir, plot
from datasets import Dataset
class tsne:
    """
    Class for t-SNE makes an object with the corresponding parameters.
    """
    def __init__(self, d_components=2, initial_dims=30, initialization='PCA', perplexity=40, dof=1., early_exaggeration=4,
                 random_state=None, data_name = '', grad_method = 'gains', max_iter =1000, initial_momentum=0.5, final_momentum=0.8, learning_rate = 500.0):
        self.d_components = d_components # dimension of projection space
        self.initial_dims = initial_dims # initial dimensions of data, before applying t-sne
        self.initialization = initialization # initialization method if there is one, defaults to PCA and uses initial_dims
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.max_iter = max_iter
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.learning_rate = learning_rate
        self.grad_method = grad_method
        self.data_name = data_name
        self.dof = dof

    def grad_descent_gains(self, X, Y, P):
        '''
        Gradient descent according to the method described in (Maaten & Hinton 2008)
        Y is the initial solution
        '''
        P = P * self.early_exaggeration
        (n, d) = X.shape
        cost = np.zeros(self.max_iter)
        min_gain = 0.01

        dY = np.zeros((n, self.d_components)) # gradient
        iY = np.zeros((n, self.d_components))# used for momemntum
        gains = np.ones((n, self.d_components)) # adaptive
        mean_abs_dY = np.zeros((self.max_iter, self.d_components))
        t0 = time()
        for iter in range(self.max_iter):
            Q, num = joint_Q(Y, self.dof)

            PQ_diff = P - Q
            # gradient:
            for i in range(n):
                dY[i, :] = ((2.*self.dof+2.)/self.dof)*np.sum(np.tile(PQ_diff[:, i] * num[:, i], (self.d_components, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = self.initial_momentum
            else:
                momentum = self.final_momentum
            # individual gains
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - self.learning_rate * (gains * dY)
            Y = Y + iY
            #Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute cost function
            cost[iter] = np.sum(P * np.log(P / Q))
            mean_abs_dY[iter, :] = np.mean(np.abs(dY), axis=0)
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration: %d cost: %.4f Mean Absolute gradient value: %s elapsed time: %.2f " % (iter + 1, C, str(mean_abs_dY[iter,:]),time() - t0))

            # Stop the early exaggeration
            if iter == 100:
                P = P / self.early_exaggeration
        np.savetxt('Models/probs/COIL20dim2gains.csv', Q, delimiter=',')
        return Y, cost, mean_abs_dY

    def grad_descent_ADAM(self, X, Y, P ):
        '''
        Gradient descent according to ADAM
        Y is the initial solution
        '''

        P = P * self.early_exaggeration
        (n, d) = X.shape
        cost = np.zeros(self.max_iter)
        dY = np.zeros((n, self.d_components)) # gradient
        mean_abs_dY = np.zeros((self.max_iter, self.d_components))
        alpha = self.learning_rate
        beta_1 = 0.85
        beta_2 = 0.9  # initialize the values of the parameters
        epsilon = 1e-8

        m_t = np.zeros((n, self.d_components))# first moment
        v_t = np.zeros((n, self.d_components))# second moment
        t0 = time()
        for iter in range(self.max_iter):
            t = iter + 1
            Q, num = joint_Q(Y, self.dof)
            Q = np.maximum(Q, 1e-12)
            PQ_diff = P - Q
            # gradient:
            for i in range(n):
                dY[i, :] = ((2.*self.dof+2.)/self.dof) * np.sum(np.tile(PQ_diff[:, i] * num[:, i], (self.d_components, 1)).T * (Y[i, :] - Y), 0)
            m_t = beta_1 * m_t + (1 - beta_1) * dY
            v_t = beta_2 * v_t + (1 - beta_2) * (dY * dY)
            m_corr = m_t / (1 - (beta_1 ** t))
            v_corr = v_t / (1 - (beta_2 ** t))
            Y = Y - (alpha * m_corr) / (np.sqrt(v_corr) + epsilon)

            # Compute cost function
            cost[iter] = np.sum(P * np.log(P / Q))
            mean_abs_dY[iter, :] = np.mean(np.abs(dY), axis=0)
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration: %d cost: %.4f Mean Absolute gradient value: %s elapsed time: %.2f " % (iter + 1, C, str(mean_abs_dY[iter,:]),time() - t0))

            # Stop the early exaggeration
            if iter == 100:
                P = P / self.early_exaggeration
        np.savetxt('Models/probs/COIL20dim2ADAM.csv', Q, delimiter=',')
        return Y, cost, mean_abs_dY

    def grad_descent(self, X, Y, P):
        '''
        Regular gradient descent Y is the initial solution
        '''
        P = P * self.early_exaggeration
        (n, d) = X.shape
        cost = np.zeros(self.max_iter)

        dY = np.zeros((n, self.d_components)) # gradient
        mean_abs_dY = np.zeros((self.max_iter, self.d_components))

        t0 = time()
        for iter in range(self.max_iter):
            Q, num = joint_Q(Y, self.dof)

            PQ_diff = P - Q
            # gradient:
            for i in range(n):
                dY[i, :] = ((2.*self.dof+2.)/self.dof)*np.sum(np.tile(PQ_diff[:, i] * num[:, i], (self.d_components, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            Y = Y - self.learning_rate * dY
            #Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute cost function
            cost[iter] = np.sum(P * np.log(P / Q))
            mean_abs_dY[iter, :] = np.mean(np.abs(dY), axis=0)
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration: %d cost: %.4f Mean Absolute gradient value: %s elapsed time: %.2f " % (iter + 1, C, str(mean_abs_dY[iter,:]),time() - t0))

            # Stop the early exaggeration
            if iter == 100:
                P = P / self.early_exaggeration
        return Y, cost, mean_abs_dY

    @profile
    def transform(self, X):
        """
        Reduces the dimensionality of X with t-SNE according to gradient descent
        """
        print("Start transforming X...")
        begin = time()

        if self.random_state is not None:
            print("transforming X with random state: " + str(self.random_state))
            np.random.seed(self.random_state)
        else:
            print("No random state specified...")

        if(self.initialization is "PCA"):
            print("First reducing dimensions of X with PCA to %.2f dimensions" %(self.initial_dims))
            X, _ = pca(X, self.initial_dims)


        (n, d) = X.shape
        Y = np.random.randn(n, self.d_components) # initialize a random solution

        cond_P, _ = cond_probs(X, perplexity=self.perplexity)
        P = joint_average_P(cond_P)
        #np.savetxt('results/' + self.data_name + 'Probabilities'+self.grad_method + '.csv', P, delimiter=',' )

        print("Start gradient descent...")
        t0 = time()
        if self.grad_method == 'ADAM':
            Y, cost, grad_value = self.grad_descent_ADAM(X, Y, P)
        elif self.grad_method == 'gains':
            Y, cost, grad_value = self.grad_descent_gains(X, Y, P)
        elif self.grad_method == 'SGD':
            Y, cost, grad_value = self.grad_descent(X, Y, P)

        #np.savetxt('results/' + self.data_name + '/' +self.grad_method  + 'cost' + str(self.d_components) +'.csv', cost, delimiter=',' )
        #np.savetxt('results/' + self.data_name + '/'+ self.grad_method +  'Y' +str(self.d_components) +'.csv', Y, delimiter=',')


        print("Gradient descent took %.4f seconds" % (time() - t0))

        return Y, cost, grad_value




if __name__ == '__main__':
    '''
    main to implement tSNE on the datasets
    '''
    seed = 0
    dataset = Dataset(seed)
    d_components = [2]
    data_name = 'COIL20'
    grad_method = 'gains'
    n_train = 960
    X, y, X_train, y_train, X_test, y_test = dataset.get_data(data_name, n_train, 10000)
    #X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()

    for d in d_components:
        if grad_method == 'ADAM':
            model = tsne(random_state=0, initialization='PCA', initial_dims=30, grad_method='ADAM', perplexity=40,
                         max_iter=1000, d_components=d, learning_rate=0.1)
        elif grad_method =='gains':
            model = tsne(random_state=0, initialization='PCA', initial_dims=30, grad_method='gains', perplexity=40,
                     max_iter=1000, d_components=d, learning_rate=100)
        elif grad_method == 'SGD':
            model = tsne(random_state=0, initialization='PCA', initial_dims=30, grad_method='SGD', perplexity=40,
                         max_iter=1000, d_components=d, learning_rate=100)

        file_path = 'Models/tSNE/' + data_name + str(n_train) + 'dim' + str(d) + grad_method
        make_dir(file_path)
        Y, cost, grad_value = model.transform(X_train)
        make_dir(file_path)
        np.savetxt(file_path + 'Y2.csv', Y, delimiter=',')
        np.savetxt(file_path + 'cost.csv', cost, delimiter=',')
        np.savetxt(file_path + 'grad_value.csv', grad_value, delimiter=',')
        #plot(Y, y_train, cmap='Paired', s = 1, linewidth= 0.1)







