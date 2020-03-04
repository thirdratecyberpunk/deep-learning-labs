# using Python and PyTorch to implement logistic regression on the iris dataset
# logistic regression is a probabilistic model that defines the probability of x
# belonging to class t as p(t|x,W) = y_t_ where y_t_ are elements of the soft-max
# vector
# estimating the weights in the soft-max vector is achieved by minimizing the
# negative log-likelihood of the data

# SECTION 1: hard coded gradient
# need to use approximate optimisation as minima of NLL cannot be computed
# analytically
# this can be achieved by gradient descent algorithm:
# while not converged:
# compute gradient of error at w
# move w along the direction of the negative gradient by a small amount

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

# loading the dataset
iris = datasets.load_iris()
# loading images
X = iris.data
# loading labels
T = iris.target
# calculating number of possible classes
C = len(set(list(T)))
# calculating number of features in each datapoint
F = X.shape[1]

# computing single value of NLL first
x = X[0]
t = T[0]
# initialising weights of model
W = np.random.randn(C,F)
# calculating linear activations for soft-max function
p = np.exp(W @ x)
y = p/sum(p)

# confirming that this is a probability distribution by checking it sums to 1
print(y)
print(sum(y))

# defines NLL for datapoint
L = - np.log(y[t])

print(L)

# holds gradient of L with respect to weights (same dimensions as W)
dW = np.zeros_like(W)

# calculating gradient using mentioned formula
for c in range(C):
    # decrease value of weight by 1 if prediction matches target (moving down)
    # otherwise increase value of weight by the datapoint (moving up)
    dW[c] = (y[c] - 1 ) * x if c==t else y[c] * x
