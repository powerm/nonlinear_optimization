import numpy as np
from numpy import linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants of the angle condition
alpha1 = 0.001
alpha2 = 0.1
p = 1


def quadratic_function(x, function_only=False, gradient_only=False, alpha=1):
    """
    :param function_value_only:
    :param alpha:
    :param x: Two dimensional vector.
    :return: Value and gradient of the function x_1^2 + alpha * x_2^2.
    """
    # calculate function value
    factor = np.array((1, alpha))
    function_value = np.matmul(np.transpose(np.square(x)), factor)
    if function_only:
        return function_value
    # calculate gradient
    gradient = 2 * x * factor
    if gradient_only:
        return gradient
    return function_value, gradient


def quadratic_function_gradient(x, function_only=False, gradient_only=False, alpha=1):
    """
    :param function_value_only:
    :param alpha:
    :param x: Two dimensional vector.
    :return: Value and gradient of the function x_1^2 + alpha * x_2^2.
    """
    factor = np.array((1, alpha))
    # calculate gradient
    gradient = 2 * x * factor
    if function_only:
        return gradient
    # calculate hessian
    hessian = np.array([[2, 0], [0, 2 * alpha]])
    if gradient_only:
        return hessian
    return gradient, hessian


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


def globalized_newton_method(x0, function_gradient, stopping_tolerance, step_size_rule, maximum_iterations):
    """
    :param x0: Starting point (column vector).
    :param function_gradient: Function that returns the value and gradient of the objective function.
    :param stopping_tolerance: The algorithm stops if ||g(x)|| <= stopping_tolerance * min(1, ||g(x0)||)
    :param step_size_rule: Defines the step size rule that should be used.
    :param maximum_iterations: The maximum number of iterations to be made.
    :return: Stationary point of the given function after termination. All points created during execution.
    """
    iterations = 0
    gradient_x0, _ = function_gradient(x0, False)
    norm_gradient_x0 = min(1.00, linalg.norm(gradient_x0))
    gradient = gradient_x0
    xk = x0
    xs = [x0]
    # while stopping criterion not met
    while linalg.norm(gradient) > stopping_tolerance * norm_gradient_x0:
        # newton step
        gradient, hessian = function_gradient(xk)
        search_direction = solve(hessian, -gradient)
        # check if it not satisfies the angle condition variant, then choose gradient direction
        if - np.matmul(np.transpose(gradient), search_direction) < \
            min(alpha1, alpha2 * np.power(linalg.norm(gradient), p)) * linalg.norm(gradient) * linalg.norm(search_direction):
            search_direction = - gradient
        # compute step size
        directional_derivative = np.matmul(np.transpose(gradient), search_direction)
        step_size = step_size_rule(xk, search_direction, directional_derivative, function_gradient, gradient)
        # update
        xk = xk * step_size * search_direction
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
    animation.save("gradient_descent_newton.gif", writer="pillow")


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


x_result, iterates = globalized_newton_method(np.array((9, 3)), quadratic_function_gradient, 0.001, armijo, 100)
print("result for quadratic function", x_result)
plot_iterates(quadratic_function, iterates)
