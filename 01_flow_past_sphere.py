""" Flow past a sphere """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import draw
from skimage.draw import polygon
import discretisation_functions


def laplace2d(phi, v0x, dx, dy, l1norm_target):
    l1norm = 1
    while l1norm > l1norm_target:
        phin = phi.copy()
        phi[1:-1, 1:-1] = (
            dy**2 * (phin[1:-1, 2:] + phin[1:-1, :-2])
            + dx**2 * (phin[2:, 1:-1] + phin[:-2, 1:-1])
        ) / (2 * (dx**2 + dy**2))
        # boundary conditions
        phi[:, 1] = v0x * dx + phi[:, 0]
        phi[:, -1] = v0x * dx + phi[:, -2]
        phi[-1, :] = phi[-2, :]
        phi[1, :] = phi[0, :]
        l1norm = (np.sum(np.abs(phi[:]) - np.abs(phin[:]))) / np.sum(np.abs(phin[:]))

    return phi


def outer_boundary_conditions(phi, v0x, dx):
    phi[:, 1] = v0x * dx + phi[:, 0]
    phi[:, -1] = v0x * dx + phi[:, -2]
    phi[-1, :] = phi[-2, :]
    phi[1, :] = phi[0, :]
    return phi


def inner_boundary_conditions(phi, gamma):
    pass


def plot2d(x, y, p):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x, y)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.5, 2.5)
    ax.view_init(30, 225)
    ax.plot_surface(X, Y, p[:], cmap=cm.viridis)

    plt.show()


Lx = 4
Ly = 5
nx = 25
ny = 26
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

x = np.linspace(0, Lx, nx) - Lx / 2
y = np.linspace(0, Ly, ny) - Ly / 2

phi_init = np.ones((ny, nx))
v0x = 1
v0y = 0
phi_init[:, 0] = v0x * dx + phi_init[:, 1]
phi_init[:, -1] = v0x * dx + phi_init[:, -2]
phi_init[-1, :] = phi_init[-2, :]
phi_init[1, :] = phi_init[0, :]


phi = laplace2d(phi_init, v0x, dx, dy, 1e-4)
# plot2d(x, y, phi)

fig, (ax1, ax2) = plt.subplots(ncols=2)
X, Y = np.meshgrid(x, y)
m = ax1.pcolormesh(X, Y, phi, shading="Gouraud")
ax1.contour(X, Y, phi, linestyles="dotted", colors="black")
plt.colorbar(m, ax=ax1)
ax1.axis("equal")


u, v = np.gradient(phi)
ax2.quiver(X, Y, v, u)
ax2.axis("equal")


plt.show()
