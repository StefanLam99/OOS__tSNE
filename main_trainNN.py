'''
Main to train the neural networks or RBMs
'''
from datasets import Dataset
from  par_tSNE import *
from reg_tSNE import *

# initialization dataset/model
seed = 0
dataset = Dataset(seed)
model_type = 'auto' # par/reg/auto/PCA/kernel
data_name = 'MNIST' # MNIST/ COIL20
if data_name == 'COIL20':
    n_train = 960
    n_test = 480
    batch_size = 480
    im_shape = (32,32)
elif data_name == 'MNIST':
    n_train = 10000
    n_test = 5000
    batch_size = 1000
    im_shape = (28,28)
d_componentsss = [2]
train_RBM = False # only true if the RBM are not trained yet
train_NN = False
show_plot = True
noisy = False # if data should be noisy
epochs = 50
X, y , X_train, y_train, X_test, y_test = dataset.get_data(name=data_name, n_train=n_train, n_test=n_test)

# hyperparameters
thetas = [0.9] #[0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
c = 0.1

# training RBM
for d_components in d_componentsss:
    file_path_weights = 'Models/weightsRBM/' + data_name + '/TEST' + data_name + str(n_train) + 'dim' + str(d_components)
    if data_name == 'MNIST':
        layer_dim = [784, 500, 500, 2000, d_components]
    elif data_name == 'COIL20':
        layer_dim = [1024, 500, 500, 2000, d_components]

    if train_RBM:
        make_dir(file_path_weights)
        RBM = Autoencoder(layer_dims=layer_dim)
        RBM.pretrain(X_train.T, epochs=10)
        RBM.save(file_path_weights)

    # training neural network
    model = None
    file_path_NN = None
    if model_type == 'par':
        file_path_NN = 'Models/parametric/' + data_name + str(n_train) + model_type + 'dim' + str(d_components)
        model = neural_tSNE(epochs=epochs, batch_size=batch_size, lr=0.01, d_components=d_components)
        model.load_RBM(file_path_weights, layer_dim)
    elif model_type == 'auto':
        file_path_NN = 'Models/autoencoder/' + data_name + str(n_train) + model_type + 'dim' + str(d_components)
        model = neuralREG_tSNE(epochs=epochs, batch_size=batch_size, lr=0.01, theta=0, d_components=d_components)
        model.load_RBM(file_path_weights, layer_dim)

    thetas_all = [-1]
    models = [model]
    file_paths = [file_path_NN]

    if model_type == 'reg':
        models = []
        file_paths = []
        thetas_all = thetas
        for theta in thetas_all:
            file_path_NN = 'Models/regularized/' + data_name + str(n_train) + model_type + 'dim' + str(d_components) + '_' +str(theta)
            file_paths.append(file_path_NN)
            model = neuralREG_tSNE(epochs=epochs, batch_size=batch_size, lr=0.01, theta=theta, d_components=d_components)
            model.load_RBM(file_path_weights, layer_dim)
            models.append(model)

    if train_NN:
        for i, theta in enumerate(thetas_all):
            model = models[i]
            if(model_type is 'reg'):
                losses = model.train(X_train, noisy=noisy)
            else:
                losses = model.train(X_train)
            if noisy:
                model.save(file_paths[i] + 'noisy')
                np.savetxt(file_paths[i] + 'noisy'+'losses.csv', losses, delimiter=',')
            else:
                pass
                #model.save(file_paths[i])
                #np.savetxt(file_paths[i]+'losses.csv', losses, delimiter=',')

    if show_plot:
        for i, theta in enumerate(thetas_all):
            if noisy:
                models[i].load_model(file_paths[i] + 'noisy')
            else:
                models[i].load_model(file_paths[i])
            Y_train = models[i].predict(X_train)
            Y_test = models[i].predict(X_test)
            if data_name == 'MNIST' and d_components==2:
                plot(Y_train, y_train, s=1, linewidth=0.2,cmap='Paired', title= 'MNIST train ' +str(n_train)+' ' +model_type+ ' theta = ' + str(theta))
                plot(Y_test, y_test, s=1, linewidth=0.2, cmap='Paired', title='MNIST test ' + str(n_test)+' ' +model_type+ ' theta = ' + str(theta))
            elif data_name == 'COIL20' and d_components==2:
                plot(Y_train, y_train, s=1, linewidth=0.2, cmap='nipy_spectral', title='COIL20 train ' + str(n_train)+' ' +model_type+ ' theta = ' + str(theta))
                plot(Y_test, y_test, s=1, linewidth=0.2, cmap='nipy_spectral', title='COIL20 test ' + str(n_test)+' ' +model_type+ ' theta = ' + str(theta))

