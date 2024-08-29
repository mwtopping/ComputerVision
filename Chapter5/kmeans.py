import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt




def generate_sample_data(nclusters, numpoints):

    xs = []
    ys = []

    for ii in range(nclusters):
        cx = np.random.uniform()
        cy = np.random.uniform()
        sx = 0.1*np.random.uniform()
        sy = 0.1*np.random.uniform()

        for jj in range(numpoints):
            xs.append(np.random.normal(loc=cx, scale=sx))
            ys.append(np.random.normal(loc=cy, scale=sy))

    return np.array(xs), np.array(ys)



def find_clusters(xdata, ydata, k):


    bestclusters = None
    bestvariance = np.inf


    Niter = 100
    iter = range(Niter)
    lowestvariance = []
    for _ in tqdm(range(Niter)):

        start = np.random.choice(len(xdata), size=k)
        clusters = np.zeros_like(xdata)
        cxs = np.array([xdata[i] for i in start])
        cys = np.array([ydata[i] for i in start])

        totaltries = 0
        while True:
            totaltries += 1
            Nflips = 0
            for ii, (x, y) in enumerate(zip(xdata, ydata)):
                dists = np.sqrt((x-cxs)**2+(y-cys)**2)

                clustermembership = np.argmin(dists)
                if clusters[ii] != clustermembership:
                    Nflips += 1
                clusters[ii]=clustermembership

            # measure cluster variance
            totalvariance = 0
            for ii in range(k):
                inds = np.where(clusters==ii)[0]
                totalvariance += np.sum(np.abs(cxs[ii]-xdata[inds])**2 + np.abs(cys[ii]-ydata[inds])**2)

            # redefine cluster centers
            for ii in range(k):
                inds = np.where(clusters==ii)[0]
                cxs[ii] = np.average(xdata[inds])
                cys[ii] = np.average(ydata[inds])

            if Nflips == 0:
                break

            if totaltries > 100:
                break

        if totalvariance < bestvariance:
            bestclusters = clusters
            bestvariance = totalvariance


        lowestvariance.append(bestvariance)


    return bestclusters, iter, lowestvariance

if __name__ == "__main__":
    fig, ax = plt.subplots()
    nclusters = 10
    xdata, ydata = generate_sample_data(nclusters, 50)

    clusters, iter, lowestvar = find_clusters(xdata, ydata, nclusters)

    
    
    for x, y, c in zip(xdata, ydata, clusters):
        ax.scatter(x, y, c=c, vmin=0, vmax=nclusters, cmap='Set1')


    fig, ax = plt.subplots()
    print(iter, lowestvar)
    ax.plot(iter, lowestvar)
    ax.set_xscale('log')


#    ax.scatter(xdata, ydata)














    plt.show()



