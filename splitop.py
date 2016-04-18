import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

n = 2000
T = 5*n
t = 40
t_up = 25
dx = 1
v0 = 2
V = np.zeros(n)
p = 0
x = 100
K = p**2 / 2
X = np.linspace(0, n, n, dtype = 'float64')
x = X/5
sigma = np.sqrt(0.5)
# psi = np.zeros(n, dtype = 'float64')

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
	V = -barrier(n, v0, V)
	return V

def wave_func(x, t, sigma, n):
	psi = gaussian (x, t, sigma, n)
	psi = psi / np.sqrt(np.sum(psi*psi))
	print(np.sum(psi*psi))
	return psi

def update (psi, V, p):
	psi = np.fft.fft(psi*np.exp(-V*math.pi/2), norm = "ortho")
	psi = np.fft.ifft(psi*np.exp(-p**2*math.pi/2), norm = "ortho")
	return psi

V = barrier(n, v0, V)

for t in range (40, int(round(max(x)))):
	psi = wave_func(x, t, sigma, n)
	psi = update (psi, V, p)
	if t % t_up == 0:
		plt.plot(x, psi)
		plt.hold(True)

plt.show()
	