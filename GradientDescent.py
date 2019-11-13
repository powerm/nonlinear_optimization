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
    :return: Stationary point of the given function after termination.
    """
    iterations = 0
    _, gradient_x0 = function_gradient(x0, False)
    norm_gradient_x0 = min(1.00, linalg.norm(gradient_x0))
    gradient = gradient_x0
    xk = x0
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
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return xk


def plt_dynamic(fig, x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()




def plot_gradient_descent(x0, function_gradient, stopping_tolerance, step_size_rule, maximum_iterations):
    """
    Only for two dimensional functions.
    :param x0:
    :param function_gradient:
    :param stopping_tolerance:
    :param step_size_rule:
    :param maximum_iterations:
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    xs, ys = [], []

    iterations = 0
    _, gradient_x0 = function_gradient(x0, False)
    norm_gradient_x0 = min(1.00, linalg.norm(gradient_x0))
    gradient = gradient_x0
    xk = x0
    xs.append(xk[0])
    ys.append(xk[1])
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
        # update plot
        xs.append(xk[0])
        ys.append(xk[1])
        # update iteration count
        iterations += 1
        if iterations >= maximum_iterations:
            break
    return xk, xs, ys


x_result = gradient_steepest_descent(np.array((9, 3)), quadratic_function, 0.001, armijo, 1000)
print("result for quadratic function", x_result)

x_result = gradient_steepest_descent(np.array((0, -3)), rosenbrock, 0.001, armijo, 1000)
print("result for rosenbrock function", x_result)


x_result, xs, ys = plot_gradient_descent(np.array((0, -3)), rosenbrock, 0.001, armijo, 1000)

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


def contour(function):
    x = np.linspace(min(xs) - 1, max(xs) + 1, 100)
    y = np.linspace(min(ys) - 1, max(ys) + 1, 100)

    X, Y = np.meshgrid(x, y)
    Z = function([X, Y], True)

    plt.contourf(X, Y, Z, 100)
    plt.colorbar()


contour(rosenbrock)
anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True)
anim.save("gradient_descent.gif", writer="pillow")
