import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

n = 5000
V = np.zeros([n,n], dtype = 'float64')