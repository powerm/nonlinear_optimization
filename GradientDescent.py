import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def quadratic_function(x, function_value_only=False, alpha=1):
    """
    :param function_value_only:
    :param alpha:
    :param x: Two dimensional vector.
    :return: Value and gradient of the function x_1^2 + alpha * x_2^2.
    """
    factor = np.array((1, alpha))
    function_value = np.matmul(np.square(x), factor)
    if function_value_only:
        return function_value
    gradient = 2 * x * factor
    return function_value, gradient


def rosenbrock(x, function_value_only=False):
    x1 = x[0]
    x2 = x[1]
    function_value = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    if function_value_only:
        return function_value
    gradient = np.array((-400 * (x2 - x1**2) * x1 - 2 * (1 - x1),
                         200 * (x2 - x1**2)))
    return function_value, gradient


def armijo(xk, search_direction, directional_derivative, function_gradient, objective_function_value, gamma=0.001, beta=0.5):
    """
    :param xk: The current point.
    :param search_direction: The current search direction.
    :param directional_derivative: The current directional derivative in the search direction.
    :param function_gradient: Function that returns the value of the objective function.
    :param objective_function_value: The current objective function value.
    :param gamma: Constant 0 < gamma < 0.5 of Armijo condition.
    :param beta: Constant of Armijo condition.
    :return: Step size satisfying the Armijo condition.
    """
    step_size = 1
    while (function_gradient(xk + step_size * search_direction, True) > \
            objective_function_value + gamma * step_size * directional_derivative).all():
        step_size *= beta
    return step_size


def gradient_steepest_descent(x0, function_gradient, stopping_tolerance, step_size_rule, maximum_iterations):
    """
    Steepest descent method with Armijo step size rule.
    :param x0: Starting point (column vector).
    :param function_gradient: Function that returns the value and gradient of the objective function.
    :param stopping_tolerance: The algorithm stops if ||g(x)|| <= stopping_tolerance * min(1, ||g(x0)||)
    :param step_size_rule: Defines the step size rule that should be used.
    :param maximum_iterations: The maximum number of iterations to be made.
    :return: Stationary point of the given function after termination. All points created during execution.
    """
    iterations = 0
    _, gradient_x0 = function_gradient(x0, False)
    norm_gradient_x0 = min(1.00, linalg.norm(gradient_x0))
    gradient = gradient_x0
    xk = x0
    # keep track of iterates
    xs = []
    # while stopping criterion not met
    while linalg.norm(gradient) > stopping_tolerance * norm_gradient_x0:
        # compute search direction
        function_value, gradient = function_gradient(xk)
        search_direction = - gradient
        directional_derivative = np.matmul(np.transpose(gradient), search_direction)
        # compute step size
        step_size = step_size_rule(xk, search_direction, directional_derivative, function_gradient, function_value)
        # update
        xk = xk + step_size * search_direction
        # update iterates
        xs.append(xk)
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return xk, xs


def plot_iterates(function, iterates):
    """
    Visualizes the sequence of iteration points of the gradient descent method. Saves the result as a gif.
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
    animation.save("gradient_descent.gif", writer="pillow")


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
    Z = function([X, Y], True)

    plt.contourf(X, Y, Z, 100)
    plt.colorbar()


x_result, _ = gradient_steepest_descent(np.array((9, 3)), quadratic_function, 0.001, armijo, 1000)
print("result for quadratic function", x_result)

x_result, iterates = gradient_steepest_descent(np.array((0, -3)), rosenbrock, 0.001, armijo, 1000)
print("result for rosenbrock function", x_result)
plot_iterates(rosenbrock, iterates)
