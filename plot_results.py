import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
import numpy as np


def plot_scalar_flux(slab):
    slab_x_values = np.append(0, slab.dx)
    slab_x_values = np.cumsum(slab_x_values)
    for i in range(slab.num_cells-1):
        cell_x_values = slab_x_values[i:i+2]
        cell_y_values = slab.scalar_flux[:,i]
        plt.plot(cell_x_values, cell_y_values)
    plt.title('Scalar Flux')
    plt.show()


def plot_current(slab):
    slab_x_values = np.linspace(0, slab.length, slab.num_cells+1)
    for i in range(slab.num_cells-1):
        cell_x_values = slab_x_values[i:i+2]
        cell_y_values = slab.current[:,i]
        plt.plot(cell_x_values, cell_y_values)
    plt.title('Current')
    plt.show()


def do_results_things(slab):
    print('Left boundary = {0}'.format(slab.scalar_flux[0, 0]))
    print('Right boundary = {0}'.format(slab.scalar_flux[1, slab.num_cells - 1]))
    plot_scalar_flux(slab)
    plot_current(slab)


def plot_twod_scalar_flux(slab):
    x = slab.dx[0] / 2
    slab_x_values = np.array([x])
    for i in range(slab.num_x_cells-1):
        x = x + slab.dx[i]
        slab_x_values = np.append(slab_x_values, x)

    y = slab.dy[0] / 2
    slab_y_values = np.array([y])
    for i in range(slab.num_y_cells-1):
        y += slab.dy[i]
        slab_y_values = np.append(slab_y_values, y)

    X, Y = np.meshgrid(slab_x_values, slab_y_values)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, slab.scalar_flux, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
