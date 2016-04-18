import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

n = 1000
T = 5*n
t = 40
t_up = 50
dx = 1
v0 = 1
V = np.zeros(n)
p = 1
x = 100
K = p**2 / 2
X = np.linspace(0, n, n, dtype = 'float64')
x = X/5
sigma = 1
psi = np.zeros(n, dtype = 'float64')

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

def update (psi, V, p):
	psi = np.fft.fft(psi*np.exp(-V*math.pi/2), norm = "ortho")
	psi = np.fft.ifft(psi*np.exp(-p**2*math.pi/2), norm = "ortho")
	return psi

def wave_func(psi, x, t, sigma, n):
	psi = gaussian (x, t, sigma, n)
	psi2 = np.zeros
	den = np.sqrt(np.sum(psi2))
	psi = psi/den
	return psi

for t in range (0,round(T/5)):
	psi = wave_func(psi, x, t, sigma, n)
	psi = update (psi, V, p)
	if t % t_up == 0:
		plt.plot(x, psi)
		plt.hold()

plt.show()