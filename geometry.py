import numpy as np


def boundary_condition_validity(boundary_condition):
    if boundary_condition == 'reflecting' or boundary_condition == 'vacuum' or boundary_condition == 'beam' or boundary_condition == 'isotropic':
        return True
    else:
        return False


class Slab:
    def __init__(self, num_angles, left_boundary, right_boundary, left_strength=None, right_strength=None):
        self.tolerance = 1e-6
        # Spacial parameters
        self.length = 0
        self.num_cells = 0
        self.dx = np.array([])
        self.region = []
        self.region_boundaries = np.array([])
        self.num_regions = 0
        # Angular parameters
        self.num_angles = num_angles
        self.mu, self.weight = np.polynomial.legendre.leggauss(num_angles)
        # Cell-wise information
        self.fixed_source = np.zeros((2,0))
        self.total_xs = None
        self.scatter_xs = None
        self.scalar_flux = np.zeros((2,0))
        self.current = np.zeros((2,0))
        # Boundary definitions
        self.left_boundary_condition = left_boundary
        self.right_boundary_condition = right_boundary
        self.left_incident_strength = left_strength
        self.right_incident_strength = right_strength
        self.left_boundary = np.ones(self.num_angles)
        self.right_boundary = np.zeros(self.num_angles)

        assert boundary_condition_validity(self.left_boundary_condition) is True, "Unreadable left boundary condition"
        assert boundary_condition_validity(self.right_boundary_condition) is True, "Unreadable right boundary condition"

    def implement_left_boundary_condition(self, angle):
        if self.left_boundary_condition == 'reflecting':
            corresponding_negative_angle = np.where(self.mu == -self.mu[angle])[0][0]
            self.left_boundary[angle] = self.left_boundary[corresponding_negative_angle]
        elif self.left_boundary_condition == 'vacuum':
            self.left_boundary[:] = 0
        elif self.left_boundary_condition == 'beam':
            forwardmost_mu = np.argmax(self.mu)
            self.left_boundary[forwardmost_mu] = self.left_incident_strength
        elif self.left_boundary_condition == 'isotropic':
            # Assumes angles are listed from -1 to 1
            half_num_angles = int(self.num_angles / 2)
            self.left_boundary[half_num_angles:self.num_angles] = self.left_incident_strength
        return self.left_boundary[angle]

    def implement_right_boundary_condition(self, angle):
        if self.right_boundary_condition == 'reflecting':
            corresponding_positive_angle = np.where(self.mu == -self.mu[angle])[0][0]
            self.right_boundary[angle] = self.right_boundary[corresponding_positive_angle]
        elif self.right_boundary_condition == 'vacuum':
            self.right_boundary[:] = 0
        elif self.right_boundary_condition == 'beam':
            backwardmost_mu = np.argmin(self.mu)
            self.right_boundary[backwardmost_mu] = self.right_incident_strength
        elif self.right_boundary_condition == 'isotropic':
            half_num_angles = int(self.num_angles / 2)
            self.right_boundary[0:half_num_angles] = self.right_incident_strength
        else:
            print('Incorrect boundary conditions')

        return self.right_boundary[angle]

    def create_region(self, material_type, length, num_cells, x_left, total_xs, scatter_xs, source):
        def truncate_final_cell():
            all_but_final_cell = np.sum(self.dx[0:self.num_cells-1])
            self.dx[self.num_cells-1] = self.length - all_but_final_cell

        num_cells = int(num_cells)
        self.length += length
        self.num_regions += 1
        self.region.append(Region(material_type, length, num_cells, x_left, total_xs, scatter_xs))
        self.region_boundaries = np.append(self.region_boundaries, self.length)
        self.num_cells += num_cells
        self.scalar_flux = np.append(self.scalar_flux, np.zeros((2, num_cells)), axis=1)
        self.fixed_source = np.append(self.fixed_source, np.ones((2, num_cells))*source, axis=1)
        self.current = np.append(self.current, np.zeros((2, num_cells)), axis=1)
        self.dx = np.append(self.dx, self.region[self.num_regions-1].dx)
        if np.sum(self.dx) != self.length:
            truncate_final_cell()

    def create_material_data_arrays(self):
        self.total_xs = np.zeros(self.num_cells)
        self.scatter_xs = np.zeros(self.num_cells)

        x = self.dx[0] / 2
        for i in range(self.num_cells):
            which_region = np.searchsorted(self.region_boundaries, x)
            self.total_xs[i] = self.region[which_region].total_xs
            self.scatter_xs[i] = self.region[which_region].scatter_xs
            x = x + self.dx[i]

    def scattering_source_contribution(self):
        scattering_source = self.scatter_xs * self.scalar_flux
        return scattering_source

    def perform_angular_flux_sweep(self, angle, total_source):
        def solve_forward_linear_system():
            a_coeff = 2 * self.total_xs[cell] * self.dx[cell] + 3 * self.mu[angle]
            b_coeff = self.total_xs[cell] * self.dx[cell] + 3 * self.mu[angle]
            c_coeff = self.dx[cell] * (2*QL[cell] + QR[cell]) + 6 * self.mu[angle] * psi_left_edge
            d_coeff = self.total_xs[cell] * self.dx[cell] - 3 * self.mu[angle]
            e_coeff = 2 * self.total_xs[cell] * self.dx[cell] + 3 * self.mu[angle]
            f_coeff = self.dx[cell] * (QL[cell] + 2*QR[cell])

            coefficient_matrix = np.array([[a_coeff, b_coeff], [d_coeff, e_coeff]])
            rhs = np.array([c_coeff, f_coeff])
            LD_angular_flux = np.linalg.solve(coefficient_matrix, rhs)
            return LD_angular_flux

        def solve_backward_linear_system():
            a_coeff = 2 * self.total_xs[cell] * self.dx[cell] - 3 * self.mu[angle]
            b_coeff = self.total_xs[cell] * self.dx[cell] + 3 * self.mu[angle]
            c_coeff = self.dx[cell] * (2*QL[cell] + QR[cell])
            d_coeff = self.total_xs[cell] * self.dx[cell] - 3 * self.mu[angle]
            e_coeff = 2 * self.total_xs[cell] * self.dx[cell] - 3 * self.mu[angle]
            f_coeff = self.dx[cell] * (QL[cell] + 2*QR[cell]) - 6 * self.mu[angle] * psi_right_edge

            coefficient_matrix = np.array([[a_coeff, b_coeff], [d_coeff, e_coeff]])
            rhs = np.array([c_coeff, f_coeff])
            LD_angular_flux = np.linalg.solve(coefficient_matrix, rhs)
            return LD_angular_flux

        one_direction_angular_flux = np.zeros((2, self.num_cells))
        QL = total_source[0,:]
        QR = total_source[1,:]
        if self.mu[angle] > 0:
            psi_left_edge = self.implement_left_boundary_condition(angle)
            for cell in range(self.num_cells):
                one_direction_angular_flux[:, cell] = solve_forward_linear_system()
                psi_left_edge = one_direction_angular_flux[1, cell]         # Left edge for next cell is right edge of current cell
            self.right_boundary[angle] = one_direction_angular_flux[1, self.num_cells-1]
        else:
            psi_right_edge = self.implement_right_boundary_condition(angle)
            for cell in reversed(range(self.num_cells)):
                one_direction_angular_flux[:, cell] = solve_backward_linear_system()
                psi_right_edge = one_direction_angular_flux[0, cell]         # Right edge for next cell is left edge of current cell
            self.left_boundary[angle] = one_direction_angular_flux[0, 0]
        return one_direction_angular_flux


class Region:
    def __init__(self, material_type, length, num_cells, x_left, total_xs, scatter_xs):
        self.material = material_type
        self.length = length
        self.num_cells = int(num_cells)
        self.dx = np.ones(self.num_cells) * (self.length / self.num_cells)
        self.left_edge = x_left
        self.right_edge = self.left_edge + self.length
        self.total_xs = total_xs
        self.scatter_xs = scatter_xs
