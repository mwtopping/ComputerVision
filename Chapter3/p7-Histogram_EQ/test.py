import numpy as np
import matplotlib.pyplot as plt




xs = np.linspace(0, 1, 100000)



def f(t):

    arr = []
    
    delta = 6.0/29
    for x in t:
        if x > delta**3:
            arr.append(x**0.3333)
        else:
            arr.append(x/(3+delta**2)+2*delta/3)
    return arr



fig, ax = plt.subplots()


ax.plot(xs, f(xs))

plt.show()
