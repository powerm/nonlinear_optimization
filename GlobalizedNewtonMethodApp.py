import GlobalizedNewtonMethod as GNM
import StepSizeRules
import numpy as np
import Examples
import Plotting

x_result, iterates = GNM.globalized_newton_method(np.array((9, 3)), Examples.quadratic_function_gradient, 0.001,
                                                  StepSizeRules.armijo, 100)
print("result for quadratic function", x_result)
Plotting.plot_iterates("globalized_newton_method", Examples.quadratic_function, iterates)

