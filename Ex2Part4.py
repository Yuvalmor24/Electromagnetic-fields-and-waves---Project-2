import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

lamb = 486 * 10 ** -9
N = 500
a = 54 * 10 ** -6
L = 50 * a
f = 9 * 6 * (10 ** -3)
l = L/N

i = complex(0,1)

ifft = np.fft.ifft2
fft = np.fft.fft2
shift = np.fft.fftshift
# define sample range
n = np.arange(-np.floor(N/2), np.floor(N/2))

x = n * l # from -L/2 to L/2
y = n * l # from -L/2 to L/2
nu_x = n/L # from -1/2l to 1/2l
nu_y = n/L # from -1/2l to 1/2l

X, Y = np.meshgrid(x, y)
NU_X, NU_Y = np.meshgrid(nu_x, nu_y)



E = np.zeros((N,N), dtype="complex128")
for j in range(-2,3):
    E += (Y < (a/2)) * (Y > -a/2) * (X - ((a * (1 - 4 * j)) / 20) < a/20) * (X - ((a * (1 - 4 * j)) / 20) > -a/20)

E = E * 10
E_tilde =  L**2 * shift(ifft(shift(E)))
I = np.zeros((N,N))
transform_fresnel = np.zeros((N,N), dtype="complex128")

const = np.power(np.power(lamb,-2) - np.power(NU_X,2) - np.power(NU_Y,2), 0.5)
transform_fresnel = np.exp(-2*i*np.pi* (f / (N/2)) * const)
for j in range(0,int(N/2)):
    E_tilde = E_tilde * transform_fresnel
    E_j = L**-2 * shift(fft(shift(E_tilde)))
    I[:,n == (j - N/2)] = np.abs(E_j[:,n==0])


E_f_tilde = L**-2 * shift(fft(shift(E_tilde)))
E_f = E_f_tilde * np.exp((i * np.pi / (lamb * f)) * (np.power(X,2) + np.power(Y,2)))
E_tilde = L**2 * shift(ifft(shift(E_f)))
for j in range(0,int(N/2)):
    E_tilde = E_tilde * transform_fresnel
    E_j = L**-2 * shift(fft(shift(E_tilde)))
    I[:,n == j] = np.abs(E_j[:,n==0])

plt.pcolormesh(np.arange(0,N) * (2*f / N),x,I)

plt.title('$ \sqrt{I(x,y=0,z)}  [\\frac{Volt}{m}] $')
plt.xlabel("z[m], from 0 to 2f")
plt.ylabel("x[m], from -L/2 to L/2")


plt.show()
