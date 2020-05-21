from neural_tSNE import ParametricTSNE
from datasets import Dataset
from utils import plot

dataset = Dataset(0)


transformer = ParametricTSNE()
X, y, X_train, y_train, X_test, y_test = dataset.get_IRIS_data(n_train=100)
# suppose you have the dataset X
X_new = transformer.fit_transform(X_train)

plot(X_new, y_train)
# transform new dataset X2 with pre-trained model

# X2_new = transformer.transform(X2)