# OOS__tSNE

Techniques to obtain a parametric mapping for dimensionality reduction: parametric and RP t-SNE (neural networks for t-SNE), kernel t-SNE (kernel function for t-SNE), autoencoders and PCA. 


**RBM.py:** Class that trains a RBM with Bernoulli distributed binary visual and hidden nodes. Used to pretrain the weights of a layer in the autoencoder network.

**RBM_linear_hidden.py:** Identical to **RBM.py**, but with Gaussian distributed hidden nodes to model real valued data

**RBM_linear_visible.py:** Identical to **RBM.py**, but with Gaussian distributed visual nodes to model real valued input data.

**pretrain_autoencoder.py:** Class that pretrains an autoencoder network using the previous mentioned RBM classes.

**par_tSNE.py:** Class to make an object for the parametric t-SNE model, it is able to finetune an encoder network, and to predict projections from input data.

**reg_tSNE.py:** Class to make an object for the RP t-SNE model, it is able to finetune the autoencoder network, predict projections from input data, and reconstruct the input data from projections. Note that depending on the value of the trade-off parameter $\theta$, this class can also be used for parametric t-SNE, and regular autoencoders.
    
**kernel_tSNE.py:** Class to make an object for the kernel t-SNE model, it is able to train a parametric mapping with kernels, and predict projections from input data. 

**pca_tSNE:** Class to make an object for the PCA model, it is able to predict projections from input data. 

**kNN.py:** Class to train a k-nearest neighbor classifier on given input data.

**tSNE.py:** Class which implements the t-SNE algorithm with three different gradient descent methods: SGD, a-SGD with momentum, and Adam. Note that a-SGD with momentum is the same as the implementation by \citet{t-SNE}.

**kcrossfold.py:** Main to perform k-fold cross-validation to find the optimal trade-off paramerter for RP t-SNE with respect to the lowest generalization error.        
    
**main_results.py:** Main to obtain the generalization errors, and trustworthiness-continuity curves of our techniques.

**main_trainNN.py:** Main to pretrain and finetune the autoencoder, parametric t-SNE, and RP t-SNE (auto)encoder networks  

**datasets.py:** Class to make an object for a Dataset, which is able to preprocess and load several datasets.
   
**utils.py:** Contains different utility functions used for our techniques. 
