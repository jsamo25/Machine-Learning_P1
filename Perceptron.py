from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
#tol = Stopping criterion ; if none stop when loss > previous_loss - tol
#default 1e-3