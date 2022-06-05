import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
# for any other
# define L,N,l
lamb = 486 * 10 ** -9
N = 1500
a = 54 * 10 ** -6
L = 15 * a
f = 9 * 6 * (10 ** -3)
l = L/N


i = complex(0,1)


ifft = np.fft.ifft2
fft = np.fft.fft2
shift = np.fft.fftshift
# define sample range
n = np.arange(-np.floor(N/2), np.floor(N/2))

# the space and frequency samples are derived from n
x = n * l # from -L/2 to L/2
y = n * l # from -L/2 to L/2
nu_x = n/L # from -1/2l to 1/2l
nu_y = n/L # from -1/2l to 1/2l
# meshgrid
X, Y = np.meshgrid(x, y)
NU_X, NU_Y = np.meshgrid(nu_x, nu_y)


E = np.zeros((N,N), dtype="complex128")
for j in range(-2,3):
    E += (Y < (a/2)) * (Y > -a/2) * (X - ((a * (1 - 4 * j)) / 20) < a/20) * (X - ((a * (1 - 4 * j)) / 20) > -a/20)

E = E * 10
E_tilde =  L**2 * shift(ifft(shift(E)))
transform_fresnel = np.zeros((N,N), dtype="complex128")

transform_fresnel = np.exp(-2*i*np.pi*f*np.power(np.power(lamb,-2) - np.power(NU_X,2) - np.power(NU_Y,2), 0.5))

E_tilde_fresnel = E_tilde * transform_fresnel

E_f = L**-2 * shift(fft(shift(E_tilde_fresnel)))

E_f = E_f * np.exp(((i*np.pi) / (lamb * f)) * (np.power(X,2) + np.power(Y,2)))

E_f_tilde = L**2 * shift(ifft(shift(E_f)))

E_2f_tilde = E_f_tilde * transform_fresnel

E_2f = L**-2 * (shift(fft(shift(E_2f_tilde))))

#plt.imshow(np.power(np.abs(E_2f),2), 'twilight')
plt.pcolormesh(x,y,np.power(np.abs(E_2f),2))
plt.title('$ I(x, y, z = 2f) [{(\\frac{Volt}{m})}^{2}] $')
plt.xlabel("x[m] from -L/2 to L/2")
plt.ylabel("y[m] from -L/2 to L/2")


plt.show()
