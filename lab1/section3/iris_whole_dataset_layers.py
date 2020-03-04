# PyTorch includes a wide range of classes that predefine essential components
# of deep learning
# very rarely have to define loss functions in terms of exp/log etc.
# instead, can use premade building blocks that can be connected together
# instead of having a weights tensor, can use a Linear layer
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iris = datasets.load_iris()
# loading dataset as floats into PyTorch tensor
X = torch.tensor(iris.data, dtype=torch.float32).to(device)
# loading labels of datasets as longs (integers) into PyTorch tensor
T = torch.tensor(iris.target, dtype=torch.long).to(device)
# calculating number of possible classes
C = len(set(list(T)))
# calculating number of features in each datapoint
F = X.shape[1]

# can use all PyTorch layers as functions on tensors
W = torch.nn.Linear(F,C).to(device)
loss = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

# optimiser object abstracts away gradient descent code
# model parameter - gradient * learning rate
optim = torch.optim.SGD(W.parameters(), lr=5e-4)

Ls = []
for e in range(1000):
    print(f"Epoch {e}")
    optim.zero_grad()
    # this function takes the output of the linear layer and the labels
    # calculates the NLL
    L = loss(W(X), T)
    L.backward()
    optim.step()
    Ls.append(L.item())
plt.plot(Ls, 'r-')
plt.show()
