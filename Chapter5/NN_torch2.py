from matplotlib.backend_bases import transforms
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda

class CustomDataset(Dataset):

    def __init__(self, transform=None, target_transform=None):

        self.transform=transform
        self.target_transform=target_transform

        self.N = 10000

        xs = []
        ys = []

        for ii in range(self.N):
            x = np.random.normal(loc=0, scale=1)
            y = np.random.normal(loc=0, scale=1)

            dist = np.sqrt(x**2+y**2)

            if dist < 1:
                t = float(1.0)
            else:
                t = float(0.0)

            xs.append(torch.tensor([x,y]))
            ys.append(torch.tensor([t]))

        self.xs = xs

        self.ts = ys

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
#                            Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(self.tx[idx]), value=1)
#        return self.xs[idx], Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))(self.ts[idx])
        return self.xs[idx], self.ts[idx]





class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == "__main__":
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    dataset = CustomDataset()
    tdataset = CustomDataset()

    train_dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
    test_dataloader = DataLoader(tdataset, batch_size=500, shuffle=True)

    model = NN().to("cpu")

    learning_rate = 0.05
    batch_size=500
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []

    Nepoch = 1000
    for epoch in tqdm(range(Nepoch)):
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        losses.append(loss.detach().numpy())
    test_loop(test_dataloader, model, loss_fn)

    fig, ax = plt.subplots()
    ax.plot(np.arange(Nepoch)+1, losses)
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig('losses.png')
    fig, ax = plt.subplots()
    xx = np.linspace(-2, 2, 50)
    yy = np.linspace(-2, 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    result = np.zeros_like(XX)
    for ii, x in enumerate(xx):
        for jj, y in enumerate(yy):
            val = torch.from_numpy(np.array([x, y])).to(torch.float32)
            t = model(val)
                      
            result[jj,ii] = t

    c = ax.contourf(XX, YY, result, 100)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.colorbar(c)

    fig.savefig('result.png')
