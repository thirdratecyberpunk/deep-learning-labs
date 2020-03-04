# using PyTorch for logistic regression on whole dataset
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
Ls = []
W = torch.randn(C, F, requires_grad=True, dtype=torch.float32)

# calculating for the whole dataset
lr = 5e-4
for e in range(1000):
    print(f"Epoch {e}")
    Ltot = 0.0
    for x,t in zip(X,T):
        p = torch.exp(W @ x)
        y = p/sum(p)
        L = -torch.log(y[t])
        L.backward()
        Ltot += L.item()
    W.data -= lr * W.grad
    # zeroes gradient
    W.grad.data.zero_()
    Ls.append(Ltot)
plt.plot(Ls, 'r-')
plt.show()
