import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

maxP = 1000
numpoints = 10*maxP
p0 = 100
p = np.linspace(0, maxP, numpoints)
v0 = 0.00001
dt = 0.1
t_p = numpoints/5
V = np.zeros(maxP*10)
alpha =0.001

def gaussian(p, p0):
	psi = np.exp(-alpha*(p-p0)**2)
	psi = psi / np.sqrt(np.sum(psi*psi))
	return psi

def step(numpoints, v0, V):
	V[numpoints/2:] = v0
	return V

def barrier(numpoints, v0, V):
	V[numpoints/3: 2*numpoints/3] = v0
	return V

def well(numpoints, v0, V):
	V = -barrier(numpoints, v0, V)
	return V

def update(psi, p, V, dt):
	psi = np.fft.fft(psi*np.exp(-dt*math.pi*p**2), norm = "ortho")
	psi = np.fft.ifft(psi*np.exp(-dt*V*math.pi*2), norm = "ortho")
	return psi


plt.plot(p, V)
plt.hold(True)

psi = gaussian (p, p0)

for t in range (0, 8*maxP):
	psi = np.fft.fft(psi*np.exp(-dt*math.pi*p**2), norm = "ortho")
	if t%t_p == 0:
		plt.plot(p, np.imag(psi))
		plt.hold(True)
	psi = np.fft.ifft(psi*np.exp(-dt*V*math.pi*2), norm = "ortho")

plt.show()