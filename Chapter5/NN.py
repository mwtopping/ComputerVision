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


def activate(x):
    return np.log(1+np.exp(x))

def dactivate(x):
    return 1/(1+np.exp(-1*x))


def model(x, w11, b11, w12, b12, w21, w22, b3):

    top1 = x*w11+b11
    bottom1 = x*w12+b12 


    top1 = activate(top1)
    bottom1 = activate(bottom1)
    

    top2 = top1*w21
    bottom2 = bottom1*w22

    sum2 = top2+bottom2
    result = sum2+b3

    return result


def gradients(model, xs, ys, w11, b11, w12, b12, w21, w22, b3):

    dLdw11, dLdb11, dLdw12, dLdb12, dLdw21, dLdw22, dLdb3 = 0, 0, 0, 0, 0, 0, 0

    for x, y in zip(xs, ys):
        dLdb3 += -2*(y-model(x, w11, b11, w12, b12, w21, w22, b3))
        
        # The value of the model up to 
        dLdw21 += activate(x*w11+b11)*dLdb3
        dLdw22 += activate(x*w12+b12)*dLdb3

        dLdw11 += x*dactivate(x*w11+b11)*w21*dLdb3
        dLdw12 += x*dactivate(x*w12+b12)*w22*dLdb3

        dLdb11 += dactivate(x*w11+b11)*w21*dLdb3
        dLdb12 += dactivate(x*w12+b12)*w22*dLdb3

    return dLdw11, dLdb11, dLdw12, dLdb12, dLdw21, dLdw22, dLdb3




if __name__ == "__main__":

    fig, ax = plt.subplots()
    learning_rate = 0.01
    w11 = np.random.normal(loc=0, scale=1)
    w12 = np.random.normal(loc=0, scale=1)
    w21 = np.random.normal(loc=0, scale=1)
    w22 = np.random.normal(loc=0, scale=1)

    b11 = 0
    b12 = 0
    b3 = 0
        
    xs = np.array([0, .5, 1])
    ys = np.array([0, 1, 0])

    xarr = np.linspace(0, 1, 100)

    for ii in range(10000):
        dLdw11, dLdb11, dLdw12, dLdb12, dLdw21, dLdw22, dLdb3 = gradients(model, xs, ys, w11, b11, w12, b12, w21, w22, b3)

        w11 -= dLdw11*learning_rate
        b11 -= dLdb11*learning_rate
        w12 -= dLdw12*learning_rate
        b12 -= dLdb12*learning_rate
        w21 -= dLdw21*learning_rate
        w22 -= dLdw22*learning_rate
        b3  -= dLdb3*learning_rate
        print(dLdb3)

        yarr = model(xarr, w11, b11, w12, b12, w21, w22, b3)
    print(b3)
    ax.plot(xarr, yarr)

#    ax.set_ylim([-1, 2])
#    ax.set_xlim([0,1])
    plt.show()


