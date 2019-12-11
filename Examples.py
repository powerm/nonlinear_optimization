import numpy as np


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


def rosenbrock(x, function_value_only=False, gradient_value_only=False):
    x1 = x[0]
    x2 = x[1]
    function_value = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    if function_value_only:
        return function_value
    gradient = np.array((-400 * (x2 - x1**2) * x1 - 2 * (1 - x1),
                         200 * (x2 - x1**2)))
    if gradient_value_only:
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

