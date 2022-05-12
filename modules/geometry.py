import numpy as np


class Region:
    def __init__(self, material_type, length, x_left, total_xs, scatter_xs):
        self.material = material_type
        self.length = length
        self.left_edge = x_left
        self.right_edge = self.left_edge + self.length
        self.total_xs = total_xs
        self.scatter_xs = scatter_xs


class Slab:
    def __init__(self, length, num_cells, num_angles, num_regions, left_boundary, right_boundary, source):
        self.length = length
        self.num_cells = num_cells
        self.hx = self.length / self.num_cells
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)
        self.num_regions = num_regions
        self.region = []
        self.region_boundaries = np.array([])
        self.left_bound = left_boundary
        self.right_bound = right_boundary
        self.tolerance = 1e-6
        self.fixed_source = source

    def create_region(self, material_type, length, x_left, total_xs, scatter_xs):
        self.region.append(Region(material_type, length, x_left, total_xs, scatter_xs))
        self.region_boundaries = np.append(self.region_boundaries, x_left)

    def create_data_arrays(self):
        total_xs = np.zeros(self.num_cells)
        scatter_xs = np.zeros(self.num_cells)

        x = self.hx / 2
        for i in range(self.num_cells):
            which_region = np.searchsorted(self.region_boundaries, x) - 1
            total_xs[i] = self.region[which_region].total_xs
            scatter_xs[i] = self.region[which_region].scatter_xs
            x = x + self.hx
        return total_xs, scatter_xs

