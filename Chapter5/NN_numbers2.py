from matplotlib.backend_bases import transforms
from torchvision import datasets
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda, ToTensor
import sys
from torchvision.datasets import mnist






class CustomDataset(Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.read_data(train)

    def read_data(self, train):
        if train:
            self.data = mnist.read_sn3_pascalvincent_tensor("./EMNIST/train-images-idx3-ubyte")
            self.labels = mnist.read_sn3_pascalvincent_tensor("./EMNIST/train-labels-idx1-ubyte")
            self.N = len(self.data)
        else:
            self.data = mnist.read_sn3_pascalvincent_tensor("./EMNIST/t10k-images-idx3-ubyte")
            self.labels = mnist.read_sn3_pascalvincent_tensor("./EMNIST/t10k-labels-idx1-ubyte")
            self.N = len(self.data)


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
#                            Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(self.tx[idx]), value=1)
        return self.data[idx:idx+1].type(torch.float32)/256, Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, y, value=1))(self.labels[idx].type(torch.int64))





class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6, kernel_size=5, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linearlayers = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        convoutput = self.convlayers(x)
        linoutput = self.linearlayers(nn.Flatten()(convoutput))
        return linoutput

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    loss = np.inf
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == "__main__":


#    training_data = datasets.EMNIST(root="EMNIST",split='digits', train=True, download=True, transform=ToTensor())
#    device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#    )
#
    loss_fn = nn.CrossEntropyLoss()
#    loss_fn = nn.MSELoss()
#
    dataset = CustomDataset(train=True)
    testdata = CustomDataset(train=False)
#
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testdata, batch_size=64, shuffle=True)

#    layer = nn.Conv2d(1, out_channels=4, kernel_size=5)

    model = NN().to("cpu")
#
    learning_rate = 0.001
    batch_size=64
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#    loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)

    losses = []

    Nepoch = 100
    for epoch in tqdm(range(Nepoch)):
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        losses.append(loss.detach().numpy())


    test_loop(test_dataloader, model, loss_fn)
#
    print(losses)
    model.eval()
    plt.imshow(testdata.data[100])

    with torch.no_grad():
        for X, y in test_dataloader:
            digit = X[0:1]
            pred = model(X[0:1])
            plt.imshow(digit[0][0])
            print(pred.argmax().item())

            break

    fig, ax = plt.subplots()
    ax.plot(np.arange(Nepoch)+1, losses)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
#    fig.savefig('losses.png')
#    fig, ax = plt.subplots()
#    xx = np.linspace(-2, 2, 50)
#    yy = np.linspace(-2, 2, 50)
#    XX, YY = np.meshgrid(xx, yy)
#    result = np.zeros_like(XX)
#    for ii, x in enumerate(xx):
#        for jj, y in enumerate(yy):
#            val = torch.from_numpy(np.array([x, y])).to(torch.float32)
#            t = model(val)
#                      
#            result[jj,ii] = t
#
#    c = ax.contourf(XX, YY, result, 100)
#
#    ax.set_xlim([-2, 2])
#    ax.set_ylim([-2, 2])
#    plt.colorbar(c)
#
#    fig.savefig('result.png')
