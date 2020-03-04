# in previous examples, we manually calculated gradient descent using numpy
# this can become prohibitively difficult in more complex problems
# obtaining gradient of loss automatically is key aim of deep learning frameworks
# this is an implementation of the same task, but written using PyTorch for
# gradient calculation
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

iris = datasets.load_iris()
# loading dataset as floats into PyTorch tensor
X = torch.tensor(iris.data, dtype=torch.float32)
# loading labels of datasets as longs (integers) into PyTorch tensor
T = torch.tensor(iris.target, dtype=torch.long)
# calculating number of possible classes
C = len(set(list(T)))
# calculating number of features in each datapoint
F = X.shape[1]
# initialising weight matrix as a PyTorch matrix
# note requires_grad -> PyTorch stores the gradient in a .grad attribute for
# this tensor
W = torch.randn(C, F, requires_grad=True, dtype=torch.float32)

# starting with a single datapoint
x = X[0]
t = T[0]
p = torch.exp(W @ x)
y = p/sum(p)
L = -torch.log(y[t])

# magic code
# calling .backward() on a value tells PyTorch to calculate gradients for ALL
# TENSORS INVOLVED IN THIS COMPUTATION (assuming requires_grad is set to true)
# this flag must be set for all model parameters
L.backward()

print(L)
