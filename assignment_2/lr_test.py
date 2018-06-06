import numpy as np
import matplotlib.pyplot as plt
from lr_debug import *
import matplotlib.animation as animation


def get_thetas(X, Y):
    '''get thetas as the training history'''
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    thetas = []
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * (grad)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            thetas.append(theta)
            print('theta: ', theta)
        if np.linalg.norm(prev_theta - theta) < 1e-15 or i > 860000:
            print('Converged in %d iterations' % i)
            thetas.append(theta)
            print('theta: ', theta)
            break
    return thetas


def visualize_training(filename):
    """ visualize the changes of seperating line"""
    X, Y = load_data(filename)
    X_ = X[:, 1:, ]

    X_positive = X_[np.where(Y == 1)]  # split the neg and pos data point for scatter
    X_negative = X_[np.where(Y == -1)]

    thetas = get_thetas(X, Y)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    ax.scatter(X_positive[:, 0], X_positive[:, 1], c='r', label='Positive Points')
    ax.scatter(X_negative[:, 0], X_negative[:, 0], c='g', label='Negative Points')
    plt.legend(loc='lower center', shadow=True)
    plt.title(filename)
    plt.rcParams['figure.figsize'] = 5, 5

    line, = ax.plot([], [], lw=1)

    def init():
        line.set_data([], [])
        return line,

    def animate(theta):
        c = theta[0]
        a = theta[1]
        b = theta[2]

        def y(x):
            return - a / b * x - c / b
        x = np.linspace(0, 1, 1000)
        line.set_data(x, y)
        print(theta)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=thetas, init_func=init, blit=True)
    plt.show()


visualize_training('data_a.txt')
