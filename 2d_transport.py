# Pseudo-code for 2D transport

# Create instance of class Slab_2D storing length, number of cells in x and y, material info, boundary conditions, etc
# This allows for the single object to be passed into functions, rather than a ton of data
# num_angles keeps with the SN convention: quadratures with N/2 positive mu are referred to as SN quadrature; num_angles is N
slab = Slab_2D(num_angles=num_angles, left_boundary='beam', right_boundary='reflecting', top_boundary='vacuum', bottom_boundary='vacuum', left_strength=10)

# Class Slab_2D contains method create_region, which appends material data to appropriate arrays stored in slab
# Problem assumes that 2D slab is comprised of regions of variable length that span the entire height
# Allows for variable mesh-size
for i in range(number_of_materials):
    slab.create_region(material_type='name of material for reference', length, num_x_cells, num_y_cells, x_left, total_xs, scatter_xs, fixed_source)

# Transport loop - source iteration
converged = False
source_iterations = 0

scalar_flux = np.zeros((num_y_cells, num_x_cells))
while not converged:
    old_scalar_flux = scalar_flux
    source_iterations += 1

    # Assume isotropic scattering
    spherical_harmonic_constant = 1         # Could have actual value if using higher scattering order; see Lewis-Miller (A-31)
    associated_legendre_function = 1        # 0-0 Associated Legendre Function = 1; dependent on mu otherwise
    cosine_term = 1                         # cos(m*azimuthal_angle); if using higher-moments, this will need to be calculated per-azimuthal-angle
    even_spherical_harmonic = np.sqrt(spherical_harmonic_constant) * associated_legendre_function * cosine_term
    scattering_source_contribution = even_spherical_harmonic * slab.scatter_xs * scalar_flux

    old_total_source = fixed_source + scattering_source_contribution

    # mu and eta hard-coded from level-symmetric quadrature, Lewis-Miller Ch. 4-2
    num_points = slab.num_angles * (slab.num_angles+2) / 2      # Total for four octants
    # Calculate [i,j] angular flux in all quadrants using cell-edge angular fluxes
    edge_angular_flux = np.zeros((num_y_cells+1, num_x_cells+1))
    midpoint_angular_flux = np.zeros((num_y_cells, num_x_cells))
    edge_angular_flux[:, 0] = left_boundary_condition       # Every row, zeroth column
    edge_angular_flux[0, :] = bottom_boundary_condition     # Every column, zeroth row
    edge_angular_flux[:, num_x_cells] = right_boundary_condition    # Every row, last column
    edge_angular_flux[num_y_cells, :] = top_boundary_condition      # Every column, last row
    for n in range(num_points):
        if mu[n] > 0 and eta[n] > 0:
            # sweep right and up
        elif mu[n] > 0 and eta[n] < 0:
            # sweep right and down
        elif mu[n] < 0 and eta[n] > 0:
            # sweep left and up
        elif mu[n] < 0 and eta[n] < 0:
            # sweep left and down

    scalar_flux = np.sum(midpoint_angular_flux) / 4        # Consistent with LM
    current = np.sum(weights * midpoint_angular_flux * mu * np.sqrt(3)) / 4     # m0,l1 angular moment

    convergence is True if np.linalg.norm(scalar_flux - old_scalar_flux)/np.linalg.norm(scalar_flux) < tolerance

