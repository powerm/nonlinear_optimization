import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_iterates(name, function, iterates):
    """
    Visualizes the sequence of iteration points of the gradient descent method. Saves the result as a gif.
    :param name: Name of the output GIF file.
    :param function: A two-dimensional function.
    :param iterates: The iteration points corresponding to the minimization of the function.
    """
    iterates = np.array(iterates)
    xs = iterates[:, 0]
    ys = iterates[:, 1]

    fig = plt.figure()
    ax = plt.axes(xlim=(min(xs) - 1, max(xs) + 1), ylim=(min(ys) - 1, max(ys) + 1))
    line, = ax.plot([], [], lw=1)
    line.set_color("grey")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = xs[0:i]
        y = ys[0:i]
        line.set_data(x, y)
        return line,

    contour(function, xs, ys)
    animation = FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True)
    animation.save(f"{name}.gif", writer="pillow")


def contour(function, xs, ys):
    """
    Plots the landscape of a two-dimensional function.
    :param function: The two-dimensional function to plot.
    :param xs: Values on the x-axis (iterates).
    :param ys: Values on the y-axis (iterates).
    """
    x = np.linspace(min(xs) - 1, max(xs) + 1, 100)
    y = np.linspace(min(ys) - 1, max(ys) + 1, 100)

    X, Y = np.meshgrid(x, y)
    Z = function(np.array([X, Y]), True)

    plt.contourf(X, Y, Z, 100)
    plt.colorbar()

