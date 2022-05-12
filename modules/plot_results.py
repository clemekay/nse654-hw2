import matplotlib.pyplot as plt
import numpy as np


def plot_scalar_flux(slab):
    slab_x_values = np.linspace(0, slab.length, slab.num_cells+1)
    for i in range(slab.num_cells-1):
        cell_x_values = slab_x_values[i:i+2]
        cell_y_values = slab.scalar_flux[:,i]
        plt.plot(cell_x_values, cell_y_values)
    plt.show()
