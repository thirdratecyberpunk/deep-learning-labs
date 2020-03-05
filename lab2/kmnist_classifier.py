# week 2 lab task
# using classifying algorithms on image data
# in this case, using the KMNIST dataset (images of pre-hiragana kuzushiji)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# loading KMNIST using dataset class
# applying a data transformation so images are represented as tensors
# can include more complex transformations (i.e. random cropping, decolouring...)
# each datapoint consists of an image as a tensor and an associated class label
train_dataset = torchvision.datasets.KMNIST('.', train=True, download=True,
                                transform=torchvision.transforms.ToTensor())

test_dataset = torchvision.datasets.KMNIST('.', train=False, download = True,
                                transform=torchvision.transforms.ToTensor())


# creating a neural network architecture requires subclassing Module class
# defining a constructor and overloading the forward() method
# forward() describes how input is transformed into output
# backward() is handled automatically
class LogisticNet(nn.Module):
    def __init__(self):
        super(LogisticNet, self).__init__()
        # layer takes in elements of size 784 and returns an element of 10
        self.fc = nn.Linear(784, 10)

    def forward(self,x):
        # flattens the input tensor to a vector of expected size for fc
        x = x.view(-1, 784)
        # sending input through linear layer
        x = self.fc(x)
        # generating probability range for classes
        return F.log_softmax(x, dim = 1)

lognet = LogisticNet()

# loading data using dataloader classes
# a training batch consists of 64 images
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                            shuffle=True)

# these classes are accessed by iterators and they iterate through mini-batches
x,t = next(iter(train_loader))
# most PyTorch layers can already handle batches
# loss functions need to be reduced via averaging/summing before backward()
loss = nn.NLLLoss()
# optimiser object
optim_SGD = optim.SGD(lognet.parameters(), lr=0.01, momentum=0.5)

# training routine
def train(net, dataloader, optimizer, loss_fun):
    net.train()
    for x, t in dataloader:
        optimizer.zero_grad()
        L = loss_fun(net(x), t)
        L.backward()
        optimizer.step()

# testing routine
def test(net, dataloader, loss_fun):
    net.eval()
    total_L = 0
    correct = 0
    with torch.no_grad():
        for x, t in dataloader:
            out = net(x)
            total_L += loss_fun(out,t)
            # counting number of correct predictions
            _, pred = out.max(dim=1)
            correct+= (pred == t).sum()
    total_L /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f"\nTest set: Average loss {total_L}, Accuracy {accuracy}% \n")

print("Testing before training")
test(lognet, test_loader, nn.NLLLoss(reduction="sum"))

for e in range(10):
    print(f"Epoch:{e+1}/ 10. Training...")
    train(lognet, train_loader, optim_SGD, nn.NLLLoss())
    print("Testing...")
    test(lognet, test_loader, nn.NLLLoss(reduction="sum"))

# trying a different architecture: multilayer perceptron
class MLPNet(nn.Module):
    def __init__(self, hidden_size):
        super(MLPNet, self).__init__()
        # first fully connected layer takes images, outputs hidden size num
        self.fc1 = nn.Linear(784, hidden_size)
        # second fc layer takes hidden size as input, outputs the 10 classes
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self,x):
        x = x.view(-1, 784)
        # applies a rectified linear unit to the result of the first linear
        # layer to avoid saturation of the layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

mlpnet = MLPNet(256)

optim_SGD_mlp = optim.SGD(mlpnet.parameters(), lr=0.01, momentum=0.5)

print("Testing before any training ...")
test(mlpnet, test_loader,  nn.NLLLoss(reduction='sum'))

for e in range(20):
  print(f"Epoch: {e+1}/20. Training ...")
  train(mlpnet, train_loader, optim_SGD_mlp, nn.NLLLoss())
  print("Testing ...")
  test(mlpnet, test_loader,  nn.NLLLoss(reduction='sum'))

# visualising the weights in the first layer as images
filters = mlpnet.fc1.weight.view(-1, 28, 28).detach()

from torchvision.utils import make_grid

grid=make_grid(filters[0:50,None,:,:], nrow=10,normalize=True)
plt.imshow(grid.permute(1,2,0),)
plt.show()
