from modules.geometry import Slab, Region
from modules.transport import transport as perform_transport_in
from modules.plot_results import plot_scalar_flux


# Problem definition info -
# fixed source or fixed source distribution
# geometric definition
# boundary conditions
# convergence criteria
# maximum number of source iterations

# Read in global data - number of regions, convergence criteria, boundary conditions, maximum
# number of iterations

# Read in local data - mesh spacing, cross-sections, number of cells per region


def kornreich_parsons_symmetric_sub(length, cells_per_mfp, num_angles, time_steps=1):
    """Calculates the steady-state k-eigenvalue or time-dependent alpha-eigenvalue of the 9 mfp
        heterogeneous one-speed subcritical slab in Kornreich Parsons (2005).

        This input runs a steady-state version of the heterogeneous slab problem in
        Kornreich, Drew & Parsons, D.. (2005). Time–eigenvalue calculations in multi-region
        Cartesian geometry using Green’s functions. Annals of Nuclear Energy. 32. 964-985.
        10.1016/j.anucene.2005.02.004.
        using methods described in
        Ryan G. McClarren (2019) Calculating Time Eigenvalues of the Neutron Transport
        Equation with Dynamic Mode Decomposition, Nuclear Science and Engineering, 193:8,
        854-867, DOI: 10.1080/00295639.2018.1565014
    -------------------------------------------------------------------------------------------
    @param length: int
        Length of slab in mfp
    @param cells_per_mfp: int
        Number of cells per mfp. Total zones in problem = length*cells_per_mfp
    @param num_angles: int
        Number of angles over which to perform SN sweeps for angular flux
    @param time_steps: int
        Number of time steps over which to run. If == 1, runs steady-state.
    """
    fuel = SimpleMaterial('subcritical fuel', num_groups=1)
    moderator = SimpleMaterial('moderator', num_groups=1)
    absorber = SimpleMaterial('absorber', num_groups=1)
    # Slab contains problem definition
    slab = SymmetricHeterogeneousSlab(length=length, cells_per_mfp=cells_per_mfp, num_angles=num_angles,
                                      mod_mat=moderator, fuel_mat=fuel, abs_mat=absorber)
    # Create one-group problem container
    one_group = OneGroup(slab=slab)
    one_group.create_zones()
    num_steps = time_steps
    dt = 0.10
    if num_steps == 1:
        print("*************************====================")
        print("Initiating k-eigenvalue solver")
        print("*************************====================")
        # Compute k-eigenvalue using steady-state power iteration
        k = one_group.perform_power_iteration()
        print(k.new)
    else:
        # Run as time-dependent problem
        time_dependent = TimeUnit(slab=slab, group=one_group, num_steps=num_steps, dt=dt)
        time_dependent.calculate_time_dependent_flux()
        alpha_eigs = time_dependent.compute_alpha_eigenvalues()
        print("alphas = ", alpha_eigs)


def optically_thick_diffusive_problem():
    length = 10
    num_cells = 10
    num_angles = 2
    num_regions = 1
    incident_flux = 10                  # Approximate by having mu closest to 1 have the incident flux

    optically_thick_material = SimpleMaterial(material_type='optically thick', total_xs=100, scatter_xs=99.5)
    slab = Slab(length=length, num_cells=num_cells, num_angles=num_angles, num_regions=num_regions,
                left_boundary=incident_flux, right_boundary='reflecting')


def uniform_infinite_medium():
    length = 10
    num_cells = 100
    num_angles = 8
    num_regions = 1

    slab = Slab(length=length, num_cells=num_cells, num_angles=num_angles, num_regions=num_regions,
                left_boundary='reflecting', right_boundary='reflecting', source=5)
    slab.create_region(material_type='infinite uniform', length=length, x_left=0, total_xs=1, scatter_xs=0)

    perform_transport_in(slab)

    plot_scalar_flux(slab)

if __name__ == "__main__":
    uniform_infinite_medium()
