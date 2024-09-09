import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np



def generate_data(Nsamp):
    xs = []
    ys = []

    for ii in range(Nsamp):
        x = np.random.normal(loc=0, scale=1)
        y = np.random.normal(loc=0, scale=1)

        dist = np.sqrt(x**2+y**2)

        if dist < 1:
            t = 1
        else:
            t = 0

        xs.append([[x],[y]])
        ys.append(t)

    return xs, ys


def activate(x):
    return np.log(1+np.exp(x))

def dactivate(x):
    return 1/(1+np.exp(-1*x))

class NN:
    def __init__(self, dims):

        self.weights = []
        self.biases = []

        for ii in range(len(dims)-1):
            self.weights.append(np.random.normal(loc=0, scale=.5, size=(dims[ii+1], dims[ii])))

        for ii in range(len(dims)-1):
            self.biases.append(np.zeros((dims[ii+1], 1)))

    def forward(self, x):
        
        self.l1z = self.l1weights*x+self.l1bias
        self.l1output = activate(self.l1z)

        self.l2z = np.dot(self.l2weights.T, self.l1output)+self.l2bias
        self.l2output = activate(self.l2z)

        self.l3z = np.dot(self.l3weights.T, self.l2output)+self.l3bias
        self.l3output = activate(self.l3z)

        return self.l3output

    def evaluate(self, xs):
        newaout = xs
        for bias, weights in zip(self.biases, self.weights):
            z = np.dot(weights, newaout)+bias
            newaout = activate(z)

        return np.squeeze(newaout)

    def back(self, xs, ys, learning_rate):
        
        #initialize variables
        aoutputs = [xs]        
        zs = []
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]

        for bias, weights in zip(self.biases, self.weights):
            z = np.dot(weights, aoutputs[-1])+bias
            zs.append(z)
            newaout = activate(z)
            aoutputs.append(newaout)


    
        delta = (aoutputs[-1]-ys)*dactivate(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, np.transpose(aoutputs[-2]))        

        for ii in range(len(self.biases)-1):
            z = zs[-2-ii]
            dz = dactivate(z)
            delta = np.dot(self.weights[-1-ii].T, delta)*dz
            grad_b[-2-ii] = delta
            grad_w[-2-ii] = np.dot(delta, np.transpose(aoutputs[-3-ii]))

        return grad_b, grad_w


    def train_once(self, xarr, yarr, learning_rate):
        tot_grad_b = [np.zeros_like(b) for b in self.biases]
        tot_grad_w = [np.zeros_like(w) for w in self.weights]

        for x, y in zip(xarr, yarr):
            grad_b, grad_w = self.back(x, y, learning_rate)

            for ii in range(len(grad_b)):
                tot_grad_b[ii] += grad_b[ii]
                tot_grad_w[ii] += grad_w[ii]


        for ii in range(len(grad_b)):
            tot_grad_b[ii] /= len(yarr)
            tot_grad_w[ii] /= len(yarr)

        for ii in range(len(grad_b)):
            self.biases[ii] -= learning_rate*tot_grad_b[ii]
            self.weights[ii] -= learning_rate*tot_grad_w[ii]


if __name__ == "__main__":

    fig, ax = plt.subplots()
    learning_rate = 0.5

    xs, ys = generate_data(100)
    xarr = np.linspace(0, 1, 100)
    network = NN([2, 4, 1])

    print(network.evaluate(xs[0]))
    Niter = 5000
    for ii in tqdm(range(Niter)):
        network.train_once(xs, ys, learning_rate)

    xx = np.linspace(-2, 2, 50)
    yy = np.linspace(-2, 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    result = np.zeros_like(XX)
    for ii, x in enumerate(xx):
        for jj, y in enumerate(yy):
            t = network.evaluate([[x], [y]])
            result[jj,ii] = t

    ax.contourf(XX, YY, result)
    for x, y in zip(xs, ys):
        if y == 1:
            color='black'
        else:
            color='white'

        c = ax.scatter(x[0], x[1], color=color, linewidth=1, edgecolor='black')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.colorbar(c)
    plt.show()
