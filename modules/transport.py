import numpy as np


def calculate_updated_scalar_flux(total_source, slab):
    angular_flux = np.zeros((2, slab.num_cells, slab.num_angles))
    for angle in range(slab.num_angles):
        angular_flux[:, :, angle] = slab.perform_angular_flux_sweep(angle, total_source)
    scalar_flux = np.sum(slab.weight * angular_flux, axis=2)
    return scalar_flux


def check_scalar_flux_convergence(new_scalar_flux, slab):
    if np.max(np.abs(new_scalar_flux - slab.scalar_flux) / new_scalar_flux) <= slab.tolerance:
        return True
    else:
        return False


def transport(slab):
    slab.create_material_data_arrays()

    converged = False
    source_iterations = 0
    # Perform source iteration to converge on one-group scalar flux
    while not converged:
        source_iterations = source_iterations + 1

        total_source = (slab.fixed_source + slab.scattering_source_contribution()) / 2

        new_scalar_flux = calculate_updated_scalar_flux(total_source, slab)

        converged = check_scalar_flux_convergence(new_scalar_flux, slab)

        slab.scalar_flux = new_scalar_flux

    print(source_iterations)
