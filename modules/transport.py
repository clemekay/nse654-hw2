import numpy as np


def calculate_one_direction_angular_flux():
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

    one_direction_angular_flux = np.zeros((num_cells,2))
    if mu > 0:
        psi_left_edge = psi_left_incident
        for i in range(num_cells):
            inside_cell_angular_flux = solve_forward_linear_system(psi_left_edge)
            psi_left_edge = inside_cell_angular_flux[1]
            one_direction_angular_flux[i,:] = inside_cell_angular_flux
    else:
        psi_right_edge = psi_right_incident
        for i in range(num_cells):
            inside_cell_angular_flux = solve_backward_linear_system(psi_right_edge)
            psi_right_edge = inside_cell_angular_flux[0]
            one_direction_angular_flux[i,:] = inside_cell_angular_flux
    return one_direction_angular_flux


def calculate_updated_scalar_flux(total_source):
    angular_flux = np.zeros((num_angles, num_cells, 2))
    for angle in range(num_angles):
        angular_flux[angle, :, :] = calculate_one_direction_angular_flux()
    scalar_flux = np.sum(weights*midpoint_angular_fluxes, axis=0)
    return scalar_flux


def scattering_source_contribution():
    scattering_source = scattering_xs*scalar_flux
    return scattering_source


def check_scalar_flux_convergence():
    if np.linalg.norm(phi - phi_new) <= tolerance:
        return True
    else:
        return False


def transport(slab):
    total_xs, scatter_xs = slab.create_data_arrays()
    # All values will have a L and R in each cell
    scalar_flux = np.ones((slab.num_cells,2))
    fixed_source = np.ones((slab.num_cells,2))

    converged = False
    source_iterations = 0
    # Perform source iteration to converge on one-group scalar flux
    while not converged:
        source_iterations = source_iterations + 1

        total_source = (fixed_source + scattering_source_contribution()) / 2

        scalar_flux = calculate_updated_scalar_flux()

        converged = check_scalar_flux_convergence()

