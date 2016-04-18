import numpy as np
from numba import jit
import matplotlib.pyplot as plt

n = 1000
T = 5*n
t = 10
t_up = 50
dx = 1
v0 = 1
V = np.zeros(n)
p = 1
x = 100
K = p**2 / 2
X = np.linspace(0, n, n, dtype = 'float')
x = X/5
sigma = 1
psi = np.zeros(n)

def gaussian(x, t, sigma, n):
	num = (x-t)**2
	sigma2 = sigma * sigma
	return np.exp(-num/2*sigma2)

def step(n, v0, V):
	V[n/2:] = v0
	return V

def barrier(n, v0, V):
	V[n/3: 2*n/3] = v0
	return V

def well(n, v0, V):
	V = -barrier(n, V)
	return V

def update (psi, t, ):
	psi = np.fft.fft(psi, ortho, )

def wave_func(psi, x, t, sigma, n):
	psi = gaussian (x, t, sigma, n)
	psi = psi / np.sqrt(np.sum(psi**2))
	return psi

psi = wave_func(psi, x, t, sigma, n)
print(sum(psi**2))
plt.plot(X, psi)
plt.show()