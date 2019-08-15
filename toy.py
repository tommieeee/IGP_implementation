import numpy as np
import GPy
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.pylab

t = 10
i = 5
mu, sigma = i/5, 0.1
time = np.array([range(t)])
time = time.reshape(10, 1)
x_coord = np.random.normal(mu, sigma, t)
x_coord = x_coord.reshape(10, 1)
y_coord = np.random.normal(0.2, sigma, 1)
for j in range(2, t+1):
    y_coord = np.append(y_coord, np.random.normal(2*j/10, sigma, 1))
y_coord = y_coord.reshape(10, 1)

coord = np.column_stack((x_coord, y_coord))
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(time, y_coord, kernel)
#display(m)
#fig3 = m.plot()
#matplotlib.pylab.show(block=True)
#display(GPy.plotting.show(fig3, filename='basic_gp_regression_notebook'))
m.optimize(messages=True)
m.optimize_restarts(num_restarts = 10)
display(m)
fig = m.plot()
matplotlib.pylab.show(block=True)
display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized'))

testX = np.linspace(0, 10, 22).reshape(-1, 1)
posteriorTestY = m.posterior_samples_f(testX, full_cov=True, size=1)