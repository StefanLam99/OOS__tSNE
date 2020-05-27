import numpy as np
from time import time
from utils import gauss_kernel, cond_probs, pca, joint_Q, joint_average_P, profile, plot
from datasets import Dataset


class kernel_tSNE:
    def __init__(self, d_components=2, initial_dims=30, initialization='PCA', perplexity=30, dof=1., early_exaggeration=4,
                 random_state=None, data_name = '', max_iter =1000, alpha=None, lr = 0.5,beta_1 = 0.9,
                 beta_2 = 0.999  ,X_train = None):

        self.d_components = d_components # dimension of projection space
        self.initial_dims = initial_dims # initial dimensions of data, before applying t-sne
        self.initialization = initialization # initialization method if there is one, defaults to PCA and uses initial_dims
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.max_iter = max_iter
        self.data_name = data_name
        self.dof = dof
        self.alpha = alpha
        self.X_train = X_train
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        if(self.initialization is "PCA"):
            print("First reducing dimensions of X_train with PCA to %.2f dimensions" %(self.initial_dims))
            self.X_train = pca(self.X_train, self.initial_dims).real
        cond_p, self.sigma = cond_probs(self.X_train)
        self.P = joint_average_P(cond_p)

    @profile
    def train(self):
        if self.random_state is not None:
            print("training X with random state: " + str(self.random_state))
            np.random.seed(self.random_state)
        else:
            print("No random state specified...")

        # initialization
        X_train = self.X_train.copy()
        (n, D) = X_train.shape
        cost = np.zeros(self.max_iter)
        self.alpha = np.random.randn(n, self.d_components)
        # calculating kernels
        K = np.zeros((n,n)) # kernel values for all rows of x_train

        for i in range(n):
            K[i,:] = gauss_kernel(X_train[i,:], X_train, self.sigma) # n x n

        Y = np.dot(K, self.alpha) # random initial solution
        print(Y)
        # training with ADAM
        epsilon = 1e-8 # prevents division by zero
        m_t = np.zeros((n, self.d_components))
        v_t = np.zeros((n, self.d_components))
        t0 = time()
        print('Starting gradient descent...')
        for iter in range(self.max_iter):
            t = iter+1

            # calculating gradient:
            Q, num = joint_Q(Y, self.dof)

            PQ_diff = self.P - Q

            # gradient:
            dalpha = np.zeros((n, self.d_components))
            ''' 
            for l in range(n):
                for i in range(n):
                    for j in range(n):
                        dalpha[l, :] += ((2. * self.dof + 2.) / self.dof) * (PQ_diff[i, j] * num[i, j]) \
                                        * (Y[i, :] - Y[j,:])*K[i, l]
            '''
            first_term =((2. * self.dof + 2.) / self.dof) * (PQ_diff * num)

            differences = []
            for d in range(self.d_components):
                #print(Y[:,d].shape)
                differences.append(Y[:,d].reshape(-1,1) - Y[:,d].reshape(1,-1))
                difference = Y[:,d].reshape(-1,1) - Y[:,d].reshape(1,-1)
                dalpha[:,d] = np.sum(np.sum(first_term*difference,axis=1).reshape(-1,1)*K,axis=0)

            ''' 
            for l in range(n):
                kernel_factor = K[:,l].reshape(-1,1)
                for d in range(self.d_components):
                    dalpha[l, d] = np.sum(first_term * differences[d]*kernel_factor)
                    #print(dalpha)
            
            for l in range(n):
                kernel_factor = K[:,l]
                for d in range(self.d_components):
                    dalpha[l, d] = np.sum(np.sum(first_term * differences[d],axis=1)*kernel_factor)
                    #print(dalpha)
            '''

            m_t = self.beta_1 * m_t + (1 - self.beta_1) * dalpha
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * (dalpha * dalpha)
            m_corr = m_t / (1 - (self.beta_1 ** t))
            v_corr = v_t / (1 - (self.beta_2 ** t))
            self.alpha = self.alpha - (self.lr * m_corr) / (np.sqrt(v_corr) + epsilon)
            Y = np.dot(K, self.alpha)

            print('step 5: ' + str(time() - t0))
           # print(Y)
            # Compute cost function

            C = np.sum(self.P * np.log(self.P / Q))
            cost[iter] = C
            #if(iter%10==0):
            print("Iteration %d: cost is %f, elapsed time: %.2f" % (iter + 1, C, time()-t0))
            ''' 
            # Stop the early exaggeration
            if iter == 100:
                P = P / self.early_exaggeration
        '''
        print('Gradient descent took: %.2f seconds'%(time() - t0))
        return Y, self.alpha, cost


    def predict(self, X):
        if self.alpha is None:
            print('Model not yet trained!')
            return
        if(self.initialization is "PCA"):
            print("First reducing dimensions of X with PCA to %.2f dimensions" % (self.initial_dims))
            X = pca(X, self.initial_dims).real

        (m, _) = X.shape
        (n, _) = self.X_train.shape
        K = np.zeros((m,n))
        for i in range(m):
            K[i,:] = gauss_kernel(X[i,:], self.X_train, self.sigma)

        return np.dot(K, self.alpha)



if __name__ == '__main__':
    begin = time()
    seed = 0
    dataset = Dataset(seed)
    #X, y, X_train, y_train, X_test, y_test = dataset.get_IRIS_data(n_train=100)
    #X, y, X_train, y_train, X_test, y_test = dataset.get_coil20_data()
    X, y, X_train, y_train, X_test, y_test = dataset.get_MNIST_data(n_train=10000, n_test=4000)
    #alpha = np.genfromtxt('kernelCoil20alpha.csv', delimiter=',')
    model = kernel_tSNE(random_state=seed, max_iter=300, X_train=X_train)
    Y, alpha, cost = model.train()
    np.savetxt('kernelMNIST10000Y.csv', Y, delimiter=',')
    np.savetxt('kernelMNIST10000alpha.csv', alpha, delimiter=',')
    np.savetxt('kernelMNIST10000cost.csv', cost, delimiter=',')
    #np.savetxt('kernelCoil20alpha.csv', alpha, delimiter=',')
    Y = model.predict(X_train)
    #np.savetxt('test.csv' , y_test, delimiter=',')
    #np.savetxt('kerneltrainY.csv', Y, delimiter=',')
    end = time()
    plot(Y, y_train, title='Kernel t-SNE: mnist10000 train, ' )
    Y = model.predict(X_test)
    plot(Y, y_test, title='Kernel t-SNE: mnist10000 test, ' )

    '''' 
    X_train = pca(X_train, no_dims=30).real
    X_test = pca(X_test, no_dims=30).real
    _,sigma = cond_probs(X_train)
    K_train=np.zeros((1000,1000))
    np.savetxt('keneltestY.csv',Y, delimiter=',')
'''


















