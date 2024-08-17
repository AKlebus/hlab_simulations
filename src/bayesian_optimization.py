import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class BayesianOptimizationSimulator():

    def __init__(self, objective, gp = None):
        self.objective = objective
        self.n = 0
        self.X = []
        self.y = []
        self.gp = gp

    def add_data_point(self, x):
        y = float(self.objective.evaluate(x)[0])
        self.X.append(x)
        self.y.append(y)
        self.n += 1
        # if self.gp is not None:
        #     self.gp.fit(self.X[:, np.newaxis], self.y)
    
    def visualize(self):
        X = np.linspace(self.objective.bounds[0], self.objective.bounds[1], 1000)[:, np.newaxis]
        y = self.objective.evaluate_noiseless(X)

        fig, ax = plt.subplots(1, 1, figsize = (8, 6))

        ax.plot(X, y, 'k', lw = 2, zorder = 9)
        ax.scatter(self.X, self.y, c = 'r', s = 50, zorder = 10, edgecolors = (0, 0, 0))
        ax.title("Posterior (n = {})".format(self.n))
        ax.xlabel("X")
        ax.ylabel("y")
        return plt
    
    def visualize_samples(self):
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.scatter(self.X, self.y)
        ax.set_title(f"Sampled Evaluations (n = {self.n})")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_xlim(self.objective.bounds[0], self.objective.bounds[1])
        return plt
    
    def visualize_posterior(self):
        X = np.linspace(self.objective.bounds[0], self.objective.bounds[1], 1000)[:, np.newaxis]
        y = self.objective.evaluate_noiseless(X)

        fig, ax = plt.subplots(1, 1, figsize = (8, 6))

        ax.plot(X, y, 'k', lw = 2, zorder = 9)
        ax.scatter(self.X, self.y, c = 'r', s = 50, zorder = 10, edgecolors = (0, 0, 0))

        if self.gp is not None:
            y_pred, sigma = self.gp.predict(X, return_std = True)
            ax.plot(X, y_pred, 'b', lw = 2, zorder = 9)
            ax.fill_between(X.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha = 0.2, color = 'blue')
        
        ax.title("Posterior (n = {})".format(self.n))
        ax.xlabel("X")
        ax.ylabel("y")
        return plt

class ObjectiveFunction():
    """
    1D Objective functions used for Bayesian optimization.
    """

    def __init__(self, f, bounds, noise = 0):
        self.f = f
        self.bounds = bounds
        self.noise = noise
    
    def evaluate(self, x):
        res = self.f(x) + np.random.normal(0, self.noise, size = 1 if type(x) == float else x.shape[0])
        # Bound noise just for demonstration purposes
        bound = 1.5
        res = min(res, self.f(x) + bound*self.noise)
        res = max(res, self.f(x) - bound*self.noise)
        return self.f(x) + np.random.normal(0, self.noise, size = 1 if type(x) == float else x.shape[0])
    
    def evaluate_noiseless(self, x):
        return self.f(x)
    
def function_1(x):
    return -((x - 2.7) ** 2) + 5

def function_2(x):
    return 0.5 * x * np.sin(5 * x) + 0.4 * x

def function_3(x):
    return -((x - 2.7) ** 2) + 5
    