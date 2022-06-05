import matplotlib.pyplot as plt
import numpy as np

lamb = 486 * 10 ** -9
N = 2000
a = 54 * 10 ** -6
L = 32 * a
d = 0.012
l = L/N

''''
d = 1.2, L = 250 * a, F = 0.005, N = 3000
d = 0.2, L = 110 * a, F = 0.03, N = 3000
d = 0.02, L = 40 * a, F = 0.3, N = 3000
d = 0.012, L = 32 * a, F = 0.5, N = 2000 VV
d = 0.006, L = 22 * a, F = 1, N = 2000 VV
d = 0.0003, L = 20 * a, F =20, N = 4000

'''

i = complex(0,1)


fig, ax1 = plt.subplots(3,figsize = (10,15))


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
E += ((X**2 + Y**2) <= a**2)

E_tilde = L**2 * shift(ifft(shift(E)))


transform_circ = np.zeros((N,N), dtype="complex128")
transform_circ = np.exp(-2*i*np.pi*d*np.power(np.power(lamb,-2) - np.power(NU_X,2) - np.power(NU_Y,2), 0.5))

E_tilde_out = E_tilde * transform_circ

I = np.abs(L**-2 * shift(fft(shift(E_tilde_out))))
ax1[0].plot(x,np.power(I[:, n==0],2), color='r')

transform_fresnel = np.zeros((N,N), dtype="complex128")
transform_fresnel = np.exp(i * np.pi * lamb * d * (np.power(NU_X,2) + np.power(NU_Y,2))) * np.exp(-i*2*np.pi*(1/lamb))

E_tilde_fresnel = E_tilde * transform_fresnel
I_fresnel = np.abs(L**-2 * shift(fft(shift(E_tilde_fresnel))))


ax1[1].plot(x,np.power(I_fresnel[:, n==0],2), color='m')

nu_x_fran = nu_x * lamb * d
ax1[2].plot(nu_x_fran[int(5*N/12):int(7*N/12)] , np.power(np.abs(E_tilde / (lamb * d)),2)[int(5*N/12):int(7*N/12), n==0], color='g')

F = round((a**2 / (lamb*d)),3)
fig.suptitle("Fresnel Number - {}".format(F), fontweight = 'bold', fontsize = 18)
plt.xlabel(r'$x [m]$')
ax1[1].set_ylabel(r'$I(x, y = 0) [V^2 \div m^2]$')

ax1[0].set_title('Exact Formula', x = 0.13, y=1.0, pad = -14, style = 'italic')
ax1[1].set_title('Frensel Approx.', x = 0.14, y=1.0, pad=-14, style = 'italic')
ax1[2].set_title('Fraunhofer Approx.\nNotice change in x units', x = 0.17, y=0.75, pad=-14, style = 'italic')

plt.legend()
plt.savefig("Frensel - {}.png".format(F))
plt.show()


