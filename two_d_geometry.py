import numpy as np
from modules.level_symmetric_quadrature import fetch_SN_quadrature


def boundary_condition_validity(boundary_condition):
    if boundary_condition == 'reflecting' or boundary_condition == 'vacuum' or boundary_condition == 'beam' or boundary_condition == 'isotropic':
        return True
    else:
        return False


class Slab:
    def __init__(self, height, y_cells, quadrature, left_boundary, front_boundary, right_boundary, back_boundary, left_strength=None, front_strength=None, right_strength=None, back_strength=None):
        self.tolerance = 1e-6
        # Spacial parameters
        self.length = 0
        self.height = height
        self.num_x_cells = 0
        self.num_y_cells = y_cells
        self.dx = np.array([])
        self.dy = np.ones(y_cells) * height / y_cells
        self.region = []
        self.region_boundaries = np.array([])
        self.num_regions = 0
        # Angular parameters
        self.num_octant_angles = int(quadrature * (quadrature + 2) / 8)
        self.mu, self.eta, self.weight = fetch_SN_quadrature(quadrature)
        self.octant_order = np.arange(4)
        # Cell-wise information
        self.fixed_source = np.zeros((y_cells, 0))
        self.total_xs = None
        self.scatter_xs = None
        self.scalar_flux = np.zeros((y_cells, 0))
        self.current = np.zeros((y_cells, 0))
        # Boundary definitions
        self.left_boundary_condition = left_boundary
        self.front_boundary_condition = front_boundary
        self.right_boundary_condition = right_boundary
        self.back_boundary_condition = back_boundary
        self.left_incident_strength = left_strength
        self.front_incident_strength = front_strength
        self.right_incident_strength = right_strength
        self.back_incident_strength = back_strength
        self.left_boundary = np.ones((4, self.num_y_cells, self.num_octant_angles))
        self.front_boundary = np.ones((4, self.num_x_cells, self.num_octant_angles))
        self.right_boundary = np.ones((4, self.num_y_cells, self.num_octant_angles))
        self.back_boundary = np.ones((4, self.num_x_cells, self.num_octant_angles))

        assert boundary_condition_validity(self.left_boundary_condition) is True, "Unreadable left boundary condition"
        assert boundary_condition_validity(self.front_boundary_condition) is True, "Unreadable front boundary condition"
        assert boundary_condition_validity(self.right_boundary_condition) is True, "Unreadable right boundary condition"
        assert boundary_condition_validity(self.back_boundary_condition) is True, "Unreadable back boundary condition"

    def determine_octant_sweep_order(self):
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        if self.left_boundary_condition == 'reflective':
            start_mu = 'negative'
        else:
            start_mu = 'positive'
        if self.front_boundary_condition == 'reflective':
            start_eta = 'negative'
        else:
            start_eta = 'positive'
        if start_mu == 'positive':
            if start_eta == 'positive':
                self.octant_order = np.array([0, 1, 2, 3])
            elif start_eta == 'negative':
                self.octant_order = np.array([2, 0, 3, 1])
        else:
            if start_eta == 'positive':
                self.octant_order = np.array([1, 0, 3, 2])
            elif start_eta == 'negative':
                self.octant_order = np.array([3, 1, 2, 0])
        assert np.sum(self.octant_order) == 6

    def implement_left_boundary_condition(self, octant):
        # Boundary condition contains angular flux at the left boundary, per angle, per octant
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        # Left BC activated when mu is positive
        if self.left_boundary_condition == 'reflecting':
            reflected_octant = int(np.abs(octant - 3))
            self.left_boundary[octant, :, :] = self.left_boundary[reflected_octant, :, :]
        elif self.left_boundary_condition == 'vacuum':
            self.left_boundary *= 0
        elif self.left_boundary_condition == 'beam':
            # Left-incident beam is theta = 0, where mu = 1
            # This should also be where eta = 0
            most_normal_mu = np.argmax(self.mu)
            self.left_boundary[octant, :, most_normal_mu] = self.left_incident_strength
        elif self.left_boundary_condition == 'isotropic':
            # All positive mu
            # All eta
            self.left_boundary[octant, :, :] = self.left_incident_strength

    def implement_front_boundary_condition(self, octant):
        # Boundary condition contains angular flux at the left boundary, per angle, per octant
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        # Front BC activated when eta is positive
        if self.front_boundary_condition == 'reflecting':
            reflected_octant = int(np.abs(octant - 3))
            self.front_boundary[octant, :, :] = self.front_boundary[reflected_octant, :, :]
        elif self.front_boundary_condition == 'vacuum':
            self.front_boundary[:, :, :] = 0
        elif self.front_boundary_condition == 'beam':
            # Front-incident beam is omega = 0, where eta = 1
            # This should also be where mu = 0
            most_normal_eta = np.argmax(self.eta)
            self.front_boundary[octant, :, most_normal_eta] = self.front_incident_strength
        elif self.front_boundary_condition == 'isotropic':
            # All positive eta
            # All mu
            self.front_boundary[octant, :, :] = self.front_incident_strength

    def implement_right_boundary_condition(self, octant):
        # Boundary condition contains angular flux at the left boundary, per angle, per octant
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        # Right BC activated when mu is negative
        if self.right_boundary_condition == 'reflecting':
            reflected_octant = int(np.abs(octant - 3))
            self.right_boundary[octant, :, :] = self.right_boundary[reflected_octant, :, :]
        elif self.right_boundary_condition == 'vacuum':
            self.right_boundary[:, :, :] = 0
        elif self.right_boundary_condition == 'beam':
            # Right-incident beam is theta = 180, where mu = -1
            # This should also be where eta = 0
            most_normal_mu = np.argmax(self.mu)
            self.right_boundary[octant, :, most_normal_mu] = self.right_incident_strength
        elif self.right_boundary_condition == 'isotropic':
            # All negative mu
            # All eta
            self.right_boundary[octant, :, :] = self.right_incident_strength

    def implement_back_boundary_condition(self, octant):
        # Boundary condition contains angular flux at the left boundary, per angle, per octant
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        # Back BC activated when eta is negative
        if self.back_boundary_condition == 'reflecting':
            reflected_octant = int(np.abs(octant - 3))
            self.back_boundary[octant, :, :] = self.back_boundary[reflected_octant, :, :]
        elif self.back_boundary_condition == 'vacuum':
            self.back_boundary[:, :, :] = 0
        elif self.back_boundary_condition == 'beam':
            # back-incident beam is omega = 0, where eta = 1
            # This should also be where mu = 0
            most_normal_eta = np.argmax(self.eta)
            self.back_boundary[octant, :, most_normal_eta] = self.back_incident_strength
        elif self.back_boundary_condition == 'isotropic':
            # All negative eta
            # All mu
            self.back_boundary[octant, :, :] = self.back_incident_strength

    def create_region(self, material_type, length, num_cells, x_left, total_xs, scatter_xs, source):
        def truncate_final_cell():
            all_but_final_cell = np.sum(self.dx[0:self.num_x_cells - 1])
            self.dx[self.num_x_cells - 1] = self.length - all_but_final_cell

        num_cells = int(num_cells)
        self.length += length
        self.num_regions += 1
        self.region.append(Region(material_type, length, num_cells, x_left, total_xs, scatter_xs))
        self.region_boundaries = np.append(self.region_boundaries, self.length)
        self.num_x_cells += num_cells
        self.scalar_flux = np.append(self.scalar_flux, np.zeros((self.num_y_cells, num_cells)), axis=1)
        self.fixed_source = np.append(self.fixed_source, np.ones((self.num_y_cells, num_cells)) * source, axis=1)
        self.current = np.append(self.current, np.zeros((self.num_y_cells, num_cells)), axis=1)
        self.front_boundary = np.append(self.front_boundary, np.zeros((4, num_cells, self.num_octant_angles)), axis=1)
        self.back_boundary = np.append(self.back_boundary, np.zeros((4, num_cells, self.num_octant_angles)), axis=1)
        self.dx = np.append(self.dx, self.region[self.num_regions - 1].dx)
        if np.sum(self.dx) != self.length:
            truncate_final_cell()

    def create_material_data_arrays(self):
        self.total_xs = np.zeros((self.num_y_cells, self.num_x_cells))
        self.scatter_xs = np.zeros((self.num_y_cells, self.num_x_cells))

        x = self.dx[0] / 2
        for i in range(self.num_x_cells):
            which_region = np.searchsorted(self.region_boundaries, x)
            self.total_xs[:, i] = self.region[which_region].total_xs
            self.scatter_xs[:, i] = self.region[which_region].scatter_xs
            x = x + self.dx[i]

    def scattering_source_contribution(self):
        scattering_source = self.scatter_xs * self.scalar_flux
        return scattering_source
    
    def perform_angular_flux_sweep(self, octant, total_source):
        # 2D, 4 octants: pos pos (0), neg pos (1), pos neg (2), neg neg (3)
        # Using 2D diamond difference
        def solve_octant1_system():
            # i-1/2 and j-1/2 are known
            numerator = total_source[row, col] + 2*mu*i_minus_j/self.dx[col] + 2*eta*i_j_minus[col]/self.dy[row]
            denominator = self.total_xs[row, col] + 2*mu/self.dx[col] + 2*eta/self.dy[row]
            psi_ij = numerator / denominator

            i_plus = 2*psi_ij - i_minus_j
            j_plus = 2*psi_ij - i_j_minus[col, :]
            return psi_ij, i_plus, j_plus

        def solve_octant2_system():
            numerator = total_source[row, col] - 2*mu*i_plus_j/self.dx[col] + 2*eta*i_j_minus[col]/self.dy[row]
            denominator = self.total_xs[row, col] - 2*mu/self.dx[col] + 2*eta/self.dy[row]
            psi_ij = numerator / denominator

            i_minus = 2*psi_ij - i_plus_j
            j_plus = 2*psi_ij - i_j_minus[col, :]
            return psi_ij, i_minus, j_plus

        def solve_octant3_system():
            numerator = total_source[row, col] + 2*mu*i_minus_j/self.dx[col] - 2*eta*i_j_plus[col]/self.dy[row]
            denominator = self.total_xs[row, col] + 2*mu/self.dx[col] - 2*eta/self.dy[row]
            psi_ij = numerator / denominator

            i_plus = 2*psi_ij - i_minus_j
            j_minus = 2*psi_ij - i_j_plus[col, :]
            return psi_ij, i_plus, j_minus

        def solve_octant4_system():
            numerator = total_source[row, col] - 2*mu*i_plus_j/self.dx[col] - 2*eta*i_j_plus[col]/self.dy[row]
            denominator = self.total_xs[row, col] - 2*mu/self.dx[col] - 2*eta/self.dy[row]
            psi_ij = numerator / denominator

            i_minus = 2*psi_ij - i_plus_j
            j_minus = 2*psi_ij - i_j_plus[col, :]
            return psi_ij, i_minus, j_minus

        cell_angular_flux = np.zeros((self.num_y_cells, self.num_x_cells, self.num_octant_angles))
        if octant == 0:     # Left to right, front to back
            mu = self.mu
            eta = self.eta
            self.implement_left_boundary_condition(octant)
            self.implement_front_boundary_condition(octant)
            i_j_plus = np.zeros((self.num_x_cells, self.num_octant_angles))     # Need to save for next row's i_j_minus
            i_plus_j = 0
            for row in range(self.num_y_cells):
                if row == 0:
                    i_j_minus = self.front_boundary[octant, :, :] * np.ones((self.num_x_cells, self.num_octant_angles))
                else:
                    i_j_minus = i_j_plus
                for col in range(self.num_x_cells):
                    if col == 0:
                        i_minus_j = self.left_boundary[octant, row, :]
                    else:
                        i_minus_j = i_plus_j
                    cell_angular_flux[row, col, :], i_plus_j, i_j_plus[col, :] = solve_octant1_system()
                    if col == self.num_x_cells - 1:
                        self.right_boundary[octant, row, :] = i_plus_j
            self.back_boundary[octant, :, :] = i_j_plus

        elif octant == 1:       # Right to left, front to back
            mu = -self.mu
            eta = self.eta
            self.implement_right_boundary_condition(octant)
            self.implement_front_boundary_condition(octant)
            i_j_plus = np.zeros((self.num_x_cells, self.num_octant_angles))     # Need to save for next row's i_j_minus
            i_minus_j = 0
            for row in range(self.num_y_cells):
                if row == 0:
                    i_j_minus = self.front_boundary[octant, :, :] * np.ones((self.num_x_cells, self.num_octant_angles))
                else:
                    i_j_minus = i_j_plus
                for col in reversed(range(self.num_x_cells)):
                    if col == self.num_x_cells - 1:
                        i_plus_j = self.right_boundary[octant, row, :]
                    else:
                        i_plus_j = i_minus_j
                    cell_angular_flux[row, col, :], i_minus_j, i_j_plus[col, :] = solve_octant2_system()
                    if col == 0:
                        self.left_boundary[octant, row, :] = i_minus_j
            self.back_boundary[octant, :, :] = i_minus_j

        elif octant == 2:       # Left to right, back to front
            mu = self.mu
            eta = -self.eta
            self.implement_left_boundary_condition(octant)
            self.implement_back_boundary_condition(octant)
            i_j_minus = np.zeros((self.num_x_cells, self.num_octant_angles))     # Need to save for next row's i_j_minus
            i_plus_j = 0
            for row in reversed(range(self.num_y_cells)):
                if row == self.num_y_cells - 1:
                    i_j_plus = self.back_boundary[octant, :, :] * np.ones((self.num_x_cells, self.num_octant_angles))
                else:
                    i_j_plus = i_j_minus
                for col in range(self.num_x_cells):
                    if col == 0:
                        i_minus_j = self.left_boundary[octant, row, :]
                    else:
                        i_minus_j = i_plus_j
                    cell_angular_flux[row, col, :], i_plus_j, i_j_minus[col, :] = solve_octant3_system()
                    if col == self.num_x_cells - 1:
                        self.right_boundary[octant, row, :] = i_plus_j
            self.front_boundary[octant, :, :] = i_j_minus

        else:       # Right to left, back to front
            mu = -self.mu
            eta = -self.eta
            self.implement_right_boundary_condition(octant)
            self.implement_back_boundary_condition(octant)
            i_j_minus = np.zeros((self.num_x_cells, self.num_octant_angles))     # Need to save for next row's i_j_minus
            i_minus_j = 0
            for row in reversed(range(self.num_y_cells)):
                if row == self.num_y_cells - 1:
                    i_j_plus = self.back_boundary[octant, :, :] * np.ones((self.num_x_cells, self.num_octant_angles))
                else:
                    i_j_plus = i_j_minus
                for col in reversed(range(self.num_x_cells)):
                    if col == self.num_x_cells - 1:
                        i_plus_j = self.right_boundary[octant, row, :]
                    else:
                        i_plus_j = i_minus_j
                    cell_angular_flux[row, col, :], i_minus_j, i_j_plus[col, :] = solve_octant4_system()
                    if col == 0:
                        self.left_boundary[octant, row, :] = i_minus_j
            self.front_boundary[octant, :, :] = i_j_minus
        return cell_angular_flux

    def calculate_updated_scalar_flux(self, total_source):
        cell_scalar_flux = np.zeros((self.num_y_cells, self.num_x_cells))
        current = np.zeros((self.num_y_cells, self.num_x_cells))
        for i in range(4):
            octant = self.octant_order[i]
            angular_flux = self.perform_angular_flux_sweep(octant, total_source)
            cell_scalar_flux += np.sum(angular_flux * self.weight, axis=2) / 4
            current += np.sum(angular_flux * self.weight * self.mu * np.sqrt(3), axis=2) / 4
        return cell_scalar_flux, current


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
