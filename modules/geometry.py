import numpy as np


class Slab:
    def __init__(self, length, num_cells, num_angles, num_regions, left_boundary, right_boundary, source):
        # Problem parameters
        self.length = length
        self.num_cells = num_cells
        self.hx = self.length / self.num_cells
        self.num_angles = num_angles
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)
        self.num_regions = num_regions
        self.region = []
        self.region_boundaries = np.array([])
        self.tolerance = 1e-6
        # Cell-wise information
        self.fixed_source = source
        self.total_xs = None
        self.scatter_xs = None
        self.scalar_flux = np.ones((2, num_cells))
        # Boundary definitions
        self.left_boundary = np.zeros(self.num_angles)
        self.right_boundary = np.zeros(self.num_angles)
        self.define_boundaries(left_boundary, right_boundary)

    def define_boundaries(self, left, right):
        # Default defined as vacuum boundary
        if left == 'reflecting':
            # This assumes that the angles are listed by all positive, then all negative
            num_positive_angles = self.num_angles / 2
            self.left_boundary[0:num_positive_angles] = self.left_boundary[num_positive_angles:self.num_angles]
        elif left != 'vacuum':
            forwardmost_mu = np.argmax(self.mu)
            self.left_boundary[forwardmost_mu] = left
        if right == 'reflecting':
            # This assumes that the angles are listed by all positive, then all negative
            num_positive_angles = self.num_angles / 2
            self.right_boundary[num_positive_angles:self.num_angles] = self.right_boundary[0:num_positive_angles]
        elif right != 'vacuum':
            backwardmost_mu = np.argmin(self.mu)
            self.right_boundary[backwardmost_mu] = right

    def create_region(self, material_type, length, x_left, total_xs, scatter_xs):
        self.region.append(Region(material_type, length, x_left, total_xs, scatter_xs))
        self.region_boundaries = np.append(self.region_boundaries, x_left)

    def create_material_data_arrays(self):
        total_xs = np.zeros(self.num_cells)
        scatter_xs = np.zeros(self.num_cells)

        x = self.hx / 2
        for i in range(self.num_cells):
            which_region = np.searchsorted(self.region_boundaries, x) - 1
            total_xs[i] = self.region[which_region].total_xs
            scatter_xs[i] = self.region[which_region].scatter_xs
            x = x + self.hx
        return total_xs, scatter_xs

    def scattering_source_contribution(self):
        scattering_source = self.scatter_xs * self.scalar_flux
        return scattering_source

    def perform_angular_flux_sweep(self, angle):
        def solve_forward_linear_system(psi_left_edge):
            a_coeff = 2 * total_xs * dx + 3 * mu
            b_coeff = total_xs * dx + 3 * mu
            c_coeff = dx * (2 * QL + QR) + 6 * mu * psi_left_edge
            d_coeff = total_xs * dx - 3 * mu
            e_coeff = 2 * total_xs * dx + 3 * mu
            f_coeff = dx * (QL + 2 * QR)

            coefficient_matrix = np.array([[a_coeff, b_coeff], [d_coeff, e_coeff]])
            rhs = np.array([c_coeff, f_coeff])
            LD_angular_flux = np.linalg.solve(coefficient_matrix, rhs)
            return LD_angular_flux

        def solve_backward_linear_system(psi_right_edge):
            a_coeff = 2 * total_xs * dx - 3 * mu
            b_coeff = total_xs * dx + 3 * mu
            c_coeff = dx * (2 * QL + QR)
            d_coeff = total_xs * dx - 3 * mu
            e_coeff = 2 * total_xs * dx - 3 * mu
            f_coeff = dx * (QL + 2 * QR) - 6 * mu * psi_right_edge

            coefficient_matrix = np.array([[a_coeff, b_coeff], [d_coeff, e_coeff]])
            rhs = np.array([c_coeff, f_coeff])
            LD_angular_flux = np.linalg.solve(coefficient_matrix, rhs)
            return LD_angular_flux

        one_direction_angular_flux = np.zeros((2, self.num_cells))
        if self.mu[angle] > 0:
            psi_left_edge = psi_left_incident
            for i in range(num_cells):
                inside_cell_angular_flux = solve_forward_linear_system(psi_left_edge)
                psi_left_edge = inside_cell_angular_flux[1]
                one_direction_angular_flux[i, :] = inside_cell_angular_flux
        else:
            psi_right_edge = psi_right_incident
            for i in range(num_cells):
                inside_cell_angular_flux = solve_backward_linear_system(psi_right_edge)
                psi_right_edge = inside_cell_angular_flux[0]
                one_direction_angular_flux[i, :] = inside_cell_angular_flux
        return one_direction_angular_flux


class Region:
    def __init__(self, material_type, length, x_left, total_xs, scatter_xs):
        self.material = material_type
        self.length = length
        self.left_edge = x_left
        self.right_edge = self.left_edge + self.length
        self.total_xs = total_xs
        self.scatter_xs = scatter_xs

