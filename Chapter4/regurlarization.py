import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt


def phi(r):
    c = 0.6
    return np.exp(-1*r**2/c**2)

def E(data, lam, weights):
    xs, ys = data

    ED = 0

    for k in range(len(weights)):
        for l in range(len(weights)):
            ED += (weights[l]*phi(np.abs(xs[k]-xs[l]))-ys[k])**2

    EW = lam*np.sum(weights**2)

    Etot = ED+EW

    return Etot

def reconstruct(xs, weights, x):
    total = 0
    for ii in range(len(xs)):
        total += weights[ii]*phi(np.abs(x-xs[ii]))
    return total

def func(x):
    return np.sin(x)

def four_point_one():
    fig, ax = plt.subplots()
    lam = 2
    results = None
    for iter in tqdm(range(50)):
        xs = []
        ys = []
        for ii in range(20):
            x = np.random.uniform(0, 10)
            y = func(x)
            xs.append(x)
            ys.append(y)

        result = minimize(lambda w: E((xs, ys), lam, w), x0=np.zeros(len(xs)))
        curve = reconstruct(xs, result['x'], np.linspace(0, 10, 100))
        ax.plot(np.linspace(0, 10, 100), reconstruct(xs, result['x'], np.linspace(0, 10, 100)), alpha=0.2)
        if results is None:
            results = curve
        else:
            results = np.vstack((results, curve))

    ax.plot(np.linspace(0, 10, 100), func(np.linspace(0, 10, 100)))
    ax.plot(np.linspace(0, 10, 100), np.average(results, axis=0), color='black', linewidth=2)
    ax.scatter(xs, ys)
    plt.show()


def regE(data,lam, arr):
    xs, ys, xarr = data

    EDarr = [(np.interp(x, xarr, arr)-ys[ii])**2 for ii, x in enumerate(xs)]
    ED = np.sum(EDarr)
    ESarr = []
    for ii in range(len(xarr)):
        ESarr.append((arr[(ii+1+len(xarr))%len(xarr)]-arr[ii])**2)
    ES = np.sum(ESarr)
    print(ES)
    return ED+lam*ES


def four_point_two():
    fig, ax = plt.subplots()
    lam = 10
    results = None
    xs = []
    ys = []

    for ii in range(100):
        x = np.random.uniform(0, 10)
        y = func(x)
        xs.append(x)
        ys.append(y)
    lxs = np.linspace(0, 10, 50)
    result = minimize(lambda w: regE((xs, ys, lxs), lam, w), x0=np.zeros(len(lxs)))
    print(result)
#    curve = reconstruct(xs, result['x'], np.linspace(0, 10, 100))
#    ax.plot(np.linspace(0, 10, 100), curve, alpha=0.2)
    ax.plot(lxs, result['x'])
    ax.plot(np.linspace(0, 10, 100), func(np.linspace(0, 10, 100)))
    ax.scatter(xs, ys)
    plt.show()





if __name__ == "__main__":
#    four_point_one()
    four_point_two()

