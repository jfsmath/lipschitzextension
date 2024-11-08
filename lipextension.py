import numpy as np
import random
import math

# A function to compute the minimal Lipschitz constant
def lipnorm(E,f):
  #max int representation value
  max_value = 0
  for i in range(len(E)-1):
    curr_value = 0
    for j in range(dim):
      curr_value += (E[i+1][j]- E[i][j])**2
      lip_value =  np.abs((f[i+1]-f[i]) / math.sqrt(curr_value))
      if lip_value > max_value:
        max_value = lip_value
  return max_value


# distance function
# can be used on arrays
# when applied to arrays, output an array of distances
def dist(x_val,E):
  distances = np.zeros(len(x_val))
  for i, element in enumerate(x_val):
    distances[i] = np.abs(np.min(element - E))
  return distances

# formula for lower and upper lipschitz extension

def lip_ext_lower(x_val, E, f):
    L = lipnorm(E, f)
    F = np.zeros(len(x_val))
    for i, element in enumerate(x_val):
        distances = np.linalg.norm(E - element, axis=1)
        F[i] = np.max([f[j] - L * distances[j] for j in range(len(E))])
    return F

def lip_ext_upper(x_val, E, f):
    L = lipnorm(E, f)
    F = np.zeros(len(x_val))
    for i, element in enumerate(x_val):
        distances = np.linalg.norm(E - element, axis=1)
        F[i] = np.min([f[j] + L * distances[j] for j in range(len(E))])
    return F


def generate_grid(n, num_points_per_dim, range_min=-3, range_max=3):
    """
    Generates a grid of points in R^n.
    
    Parameters:
    - n: int, number of dimensions.
    - num_points_per_dim: int, number of points per dimension.
    - range_min: float, minimum value in each dimension.
    - range_max: float, maximum value in each dimension.
    
    Returns:
    - E: np.ndarray, shape (num_points_per_dim**n, n), array of points in R^n.
    """
    linspaces = [np.linspace(range_min, range_max, num_points_per_dim) for _ in range(n)]
    grid = np.meshgrid(*linspaces)
    E = np.vstack([g.ravel() for g in grid]).T
    return E
