import GlobalizedBFGS as GBFGS
import numpy as np
import Examples
import Plotting

x_result, iterates = GBFGS.globalized_BFGS(np.array((9, 3)), Examples.quadratic_function, 0.001, 100)
print("result for quadratic function", x_result)
Plotting.plot_iterates("globalized_bfgs", Examples.quadratic_function, iterates)

