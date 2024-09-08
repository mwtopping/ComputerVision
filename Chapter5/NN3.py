import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np



def generate_data(Nsamp):
    xs = []
    ys = []
    ts = []

    for ii in range(Nsamp):
        x = np.random.normal(loc=0, scale=1)
        y = np.random.normal(loc=0, scale=1)

        dist = np.sqrt(x**2+y**2)

        if dist < 1:
            t = 1
        else:
            t = 0

        xs.append(x)
        ys.append(y)
        ts.append(t)

    return xs, ys, ts


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

    def back(self, xs, ys, learning_rate):

        aoutputs = [xs]        
        zs = []
            
        for bias, weights in zip(self.biases, self.weights):
            z = np.dot(weights, aoutputs[-1])+bias
            zs.append(z)
            newaout = activate(z)
            aoutputs.append(newaout)


        print("output", aoutputs[-1])
    
        delta = (aoutputs[-1]-ys)*dactivate(zs[-1])
        print("last delta ", delta)
#        for ii, (x, y) in enumerate(zip(xs, ys)):
#            _ = self.forward(x)
#            l2deltas[ii] = np.multiply(np.dot(self.l3weights,l3deltas[ii]), dactivate(self.l2z))[0]
#            l1deltas[ii] = np.multiply(np.dot(self.l2weights,l2deltas[ii]), dactivate(self.l1z))[0]
#
#        N = len(xs)
#        grad_b1 = np.sum(l1deltas, axis=0)/N
#        grad_b2 = np.sum(l2deltas, axis=0)/N
#        grad_b3 = np.sum(l3deltas, axis=0)/N
#
#        grad_w1s = np.array([np.zeros_like(self.l1weights) for x in xs])
#        grad_w2s = np.array([np.zeros_like(self.l2weights) for x in xs])
#        grad_w3s = np.array([np.zeros_like(self.l3weights) for x in xs])
#
#        for ii in range(len(xs)):
#            grad_w3s[ii] = np.reshape(np.inner(self.l2output,l3deltas[ii]), np.shape(grad_w3s[ii]))
#            grad_w2s[ii] = np.reshape(np.inner(self.l1output,l2deltas[np.newaxis, ii].T), np.shape(grad_w2s[ii]))
#            grad_w1s[ii] = np.reshape(np.inner(xs[ii], l1deltas[ii,np.newaxis].T), np.shape(grad_w1s[ii]))
#
#        grad_w1 = np.sum(grad_w1s, axis=0)/N
#        grad_w2 = np.sum(grad_w2s, axis=0)/N
#        grad_w3 = np.sum(grad_w3s, axis=0)/N
#
#        self.l1bias -= np.reshape(grad_b1, np.shape(self.l1bias))*learning_rate
#        self.l2bias -= np.reshape(grad_b2, np.shape(self.l2bias))*learning_rate
#        self.l3bias -= np.reshape(grad_b3, np.shape(self.l3bias))*learning_rate
#
#        self.l1weights -= np.reshape(grad_w1, np.shape(self.l1weights))*learning_rate
#        self.l2weights -= np.reshape(grad_w2, np.shape(self.l2weights))*learning_rate
#        self.l3weights -= np.reshape(grad_w3, np.shape(self.l3weights))*learning_rate


if __name__ == "__main__":

    fig, ax = plt.subplots()
    learning_rate = 0.05


    xarr = np.linspace(0, 1, 100)
    network = NN([2, 4, 1])
    network.back([[0],[0]], 1, learning_rate)
