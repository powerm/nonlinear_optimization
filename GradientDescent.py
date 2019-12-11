import numpy as np
from numpy import linalg


def gradient_steepest_descent(x0, function_gradient, stopping_tolerance, step_size_rule, maximum_iterations):
    """
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
    xs = [x0]
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
