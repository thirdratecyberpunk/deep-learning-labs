# like the previous example, but for the entire iris dataset
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

# array of NLL values throughout process
Ls = []
# initial weight values
W = np.random.randn(C,F)
# learning rate
lr = 5e-4
# 1000 steps of gradient descent
for e in range(1000):
    # set gradients to zero
    dW = np.zeros_like(W)
    # initialising NLL sum
    L = 0
    # looping through dataset
    for x,t in zip(X,T):
        # calculate softmax
        p = np.exp(W @ x)
        y = p/sum(p)
        # add NLL for this datapoint
        L += -np.log(y[t])
        # compute gradient for this datapoint
        for c in range(C):
            dW[c] += (y[c] - 1) * x if c==t else y[c] * x
    # move weights in opposite direction to gradients
    W -= lr * dW
    # keep record of overall trajectory
    Ls.append(L)
# plot trajectory of L during gradient descent
plt.plot(Ls, 'r-')
plt.show()

# testing accuracy of model
correct = 0
for x,t in zip(X,T):
    a_pred = W@x
    correct += (a_pred.argmax() == t)
print(f"accuracy={correct/len(T)}")
