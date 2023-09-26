""" Flow past a sphere """

import numpy as np
import matplotlib.pyplot as plt

from skimage import draw


def create_grid(Lx, Ly, nx, ny):
    """Create rectangular grid centred on (0,0)"""
    x = np.linspace(0, Lx, nx) - Lx / 2
    y = np.linspace(0, Ly, ny) - Ly / 2
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_grid(grid):
    fig, ax = plt.subplots()
    R = np.sqrt(grid[0] ** 2 + grid[1] ** 2)
    m = ax.pcolormesh(grid[0], grid[1], R)
    plt.colorbar(m, ax=ax)
    ax.axis("equal")
    return ax


grid = create_grid(4, 5, 25, 26)
# plot_grid(grid)
# plt.show()


def parameteric_circle(num):
    t = np.linspace(0, 2 * np.pi, num, endpoint=False)
    x, y = np.cos(t), np.sin(t)
    nx, ny = x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)
    return x, y, nx, ny


def plot_circle(grid, circle):
    x, y, nx, ny = circle
    ax = plot_grid(grid)
    ax.plot(x, y, "ok")
    ax.quiver(x, y, nx, ny)


circle = parameteric_circle(35)
# plot_circle(grid, circle)
# plt.show()


def transform_to_xy(coords, grid):
    X, Y = grid
    x_disc = np.array([X[coord[0], coord[1]] for coord in coords])
    y_disc = np.array([Y[coord[0], coord[1]] for coord in coords])
    return x_disc, y_disc


def rasterisation_coordinate_setup(grid):
    X, Y = grid
    X_flat, Y_flat = X.flatten(), Y.flatten()

    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])

    C, R = np.meshgrid(c, r)
    R_flat, C_flat = R.flatten(), C.flatten()

    grid_coords = np.c_[X_flat, Y_flat]
    coords_rc = np.c_[R_flat, C_flat]

    return grid_coords, coords_rc


def discretise_points(grid_coords, coords_rc, x, y, nx=0, ny=0):
    if nx == 0:
        rcs = []
        already_seen = set()
        for xx, yy in zip(x, y):
            # calculate distance betweeen grid coordinate and given coordinates, find index of minimum distance
            dist = ((grid_coords - np.array([xx, yy]).reshape(1, -1)) ** 2).sum(axis=1)
            argmin = dist.argmin()
            if argmin not in already_seen:
                rcs.append(coords_rc[argmin])
                already_seen.add(argmin)
        return rcs
    else:
        rcs = []
        discrete_vectors = []
        already_seen = set()
        for xx, yy, nxx, nyy in zip(x, y, nx, ny):
            # calculate distance betweeen grid coordinate and given coordinates, find index of minimum distance
            dist = ((grid_coords - np.array([xx, yy]).reshape(1, -1)) ** 2).sum(axis=1)
            argmin = dist.argmin()
            if argmin not in already_seen:
                rcs.append(coords_rc[argmin])
                discrete_vectors.append([nxx, nyy])
                already_seen.add(argmin)
        return rcs, discrete_vectors


def rasterise_to_coords(x, y, grid):
    """Rasterises discrete set of (x,y) points to a grid and returns (r,c) coordinates"""
    grid_coords, coords_rc = rasterisation_coordinate_setup(grid)

    rcs = discretise_points(grid_coords, coords_rc, x, y)

    N = len(rcs)
    coords = []
    for start, stop in zip(range(N), range(1, N + 1)):
        if stop >= N:
            stop = stop % N
        r_start, c_start = rcs[start]  # take coordinate of 'start' point
        r_end, c_end = rcs[stop]  # take coordinate of 'stop' point
        rr, cc = draw.line(r_start, c_start, r_end, c_end)
        coords.extend([r, c] for r, c in zip(rr[:-1], cc[:-1]))

    return np.array(coords)


x, y, nx, ny = circle
coords = rasterise_to_coords(x, y, grid)
x_disc, y_disc = transform_to_xy(coords, grid)


ax = plot_grid(grid)
ax.plot(x_disc, y_disc, "-x")
plt.show()


circle_fine = parameteric_circle(25)
x, y, nx, ny = circle_fine
coords = rasterise_to_coords(x, y, grid)
x_disc, y_disc = transform_to_xy(coords, grid)


ax = plot_grid(grid)
ax.plot(x_disc, y_disc, "-x")
plt.show()

grid_fine = create_grid(4, 5, 55, 56)
coords = rasterise_to_coords(x, y, grid)
x_disc, y_disc = transform_to_xy(coords, grid)


ax = plot_grid(grid)
ax.plot(x_disc, y_disc, "-x")
plt.show()


def rasterise_vectors(x, y, nx, ny, grid):
    """Rasterises discrete set of (x,y) points and vectors (nx,ny) to a grid and returns (nx_disc,ny_disc) vector field."""
    grid_coords, coords_rc = rasterisation_coordinate_setup(grid)

    rcs, discrete_vectors = discretise_points(grid_coords, coords_rc, x, y, nx, ny)

    N = len(rcs)
    vector_field = []
    for start, stop in zip(range(N), range(1, N + 1)):
        if stop >= N:
            stop = stop % N
        r_start, c_start = rcs[start]
        r_end, c_end = rcs[stop]
        nx_start, ny_start = discrete_vectors[start]
        nx_end, ny_end = discrete_vectors[stop]

        rr, cc = draw.line(r_start, c_start, r_end, c_end)
        N_interp = len(rr)
        for i in range(N_interp - 1):
            alpha = i / N_interp
            vector_field.append(
                [
                    (1 - alpha) * nx_start + alpha * nx_end,
                    (1 - alpha) * ny_start + alpha * ny_end,
                ]
            )

    return np.array(vector_field)
