import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

n = 2000
T = 5*n
t = 20
t_up = 25
v0 = 3
dt = 0.001
V = np.zeros(n)
p = 10
alpha = 0.1
x0 = 10
X = np.linspace(-n, n, n, dtype = 'float64')
x = X/5
# psi = np.zeros(n, dtype = 'float64')

def gaussian(p, t, alpha, n):
	num = (x-x0)**2
	return np.exp(-alpha*num)

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
	psi_r = np.cos(x)*psi
	psi_i = np.sin(x)*psi
	return psi, psi_r, psi_i

def update (psi, psi_r, psi_i, V, p):
	psi = np.fft.fft(psi*np.exp(-dt*V*math.pi*2), norm = "ortho")
	psi = np.fft.ifft(psi*np.exp(-dt*math.pi*p**2), norm = "ortho")
	psi_r = np.fft.fft(psi_r*np.exp(-dt*V*math.pi*2), norm = "ortho")
	psi_r = np.fft.ifft(psi_r*np.exp(-dt*p**2*math.pi), norm = "ortho")
	psi_i = np.fft.fft(psi_i*np.exp(-dt*V*math.pi*2), norm = "ortho")
	psi_i = np.fft.ifft(psi_i*np.exp(-dt*p**2*math.pi), norm = "ortho")
	return psi, psi_r, psi_i

#####################################################################
########################  MAIN  #####################################
#####################################################################


V = well(n, v0, V)

for t in range (40, int(round(max(x)))):
	psi, psi_r, psi_i = wave_func(x, t, alpha, n)
	psi, psi_r, psi_i = update (psi, psi_r, psi_i, V, p)
	if t % t_up == 0:
		plt.plot(x, psi)
		plt.show()

# plt.show()