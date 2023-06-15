import numpy as np


def calculate_updated_scalar_flux(total_source, slab):
    scalar_flux = np.zeros((4, slab.num_xcells, slab.num_ycells))
    current = np.zeros((4, slab.num_xcells, slab.num_ycells))
    for ordinate in range(slab.num_octant_ordinates):
        # Sweep right and up
        angular_flux = slab.perform_angular_flux_sweep(ordinate, 1, 1, total_source)
        scalar_flux += slab.weight[ordinate] * angular_flux
        # Sweep left and up
        angular_flux = slab.perform_angular_flux_sweep(ordinate, -1, 1, total_source)
        scalar_flux += slab.weight[ordinate] * angular_flux
        # Sweep left and down
        angular_flux = slab.perform_angular_flux_sweep(ordinate, -1, -1, total_source)
        scalar_flux += slab.weight[ordinate] * angular_flux
        # Sweep right and down
        angular_flux = slab.perform_angular_flux_sweep(ordinate, 1, -1, total_source)
        scalar_flux += slab.weight[ordinate] * angular_flux
    return scalar_flux, current


def transport(slab):
    slab.create_material_data_arrays()

    converged = False
    source_iterations = 0
    # Perform source iteration to converge on one-group scalar flux
    while not converged:
        print(source_iterations)

        total_source = (slab.fixed_source + slab.scattering_source_contribution()) / 4

        new_scalar_flux, current = calculate_updated_scalar_flux(total_source, slab)

        converged = slab.check_scalar_flux_convergence(new_scalar_flux)

        slab.scalar_flux = new_scalar_flux
        slab.current = current

        source_iterations = source_iterations + 1

    print('Converged in ' + str(source_iterations-1) + ' iterations')

