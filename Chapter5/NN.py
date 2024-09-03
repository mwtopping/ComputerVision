import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    xs, ys, ts = generate_data(500)

    fig, ax = plt.subplots()

    ax.scatter(xs, ys, c=ts, cmap='coolwarm')


    plt.show()

