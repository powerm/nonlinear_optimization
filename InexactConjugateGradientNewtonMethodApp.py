import InexactConjugateGradientNewtonMethod as ICGNM
import StepSizeRules
import numpy as np
import Examples
import Plotting


x_result, iterates = ICGNM.general_descent_inexact_newton_conjugate_gradient(np.array((10, 20)),
                                                                             Examples.quadratic_function_gradient_hessian,
                                                                             0.001, 1000)
print("result for quadratic function", x_result)
Plotting.plot_iterates("inexact_newton_conjugate_gradient", Examples.quadratic_function, iterates)
