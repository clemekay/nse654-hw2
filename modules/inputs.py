from modules.geometry import Slab, Region
from modules.transport import transport as perform_transport_in
from modules.plot_results import plot_scalar_flux, plot_current


# Problem definition info -
# fixed source or fixed source distribution
# geometric definition
# boundary conditions
# convergence criteria
# maximum number of source iterations

# Read in global data - number of regions, convergence criteria, boundary conditions, maximum
# number of iterations

# Read in local data - mesh spacing, cross-sections, number of cells per region


def uniform_infinite_medium():
    length = 10
    num_cells = 100
    num_angles = 8

    slab = Slab(num_angles=num_angles, left_boundary='reflecting', right_boundary='reflecting', source=10)
    slab.create_region(material_type='infinite uniform', length=length, num_cells=num_cells, x_left=0, total_xs=2,
                       scatter_xs=0)

    perform_transport_in(slab)

    plot_scalar_flux(slab)
    plot_current(slab)


def source_free_pure_absorber():
    length = 10
    l1 = 3
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary=10, right_boundary='reflecting', source=0)
    slab.create_region(material_type='infinite uniform', length=l1, num_cells=30, x_left=0, total_xs=1, scatter_xs=0)
    slab.create_region(material_type='infinite heterog', length=length - l1, num_cells=70, x_left=l1, total_xs=1,
                       scatter_xs=0)

    perform_transport_in(slab)

    plot_scalar_flux(slab)
    plot_current(slab)


def source_free_half_space():
    length = 20
    l1 = 8
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='isotropic', right_boundary='isotropic', source=0,
                left_strength=10, right_strength=6)
    slab.create_region(material_type='left', length=l1, num_cells=80, x_left=0, total_xs=3, scatter_xs=1.5)
    slab.create_region(material_type='right', length=length - l1, num_cells=120, x_left=l1, total_xs=1, scatter_xs=0.9)

    perform_transport_in(slab)

    print('Left boundary = {0}'.format(slab.scalar_flux[0, 0]))
    print('Right boundary = {0}'.format(slab.scalar_flux[1, slab.num_cells - 1]))
    plot_scalar_flux(slab)


def optically_thick_diffusive_problem():
    length = 10
    num_cells = 100
    num_angles = 8

    slab = Slab(num_angles=num_angles, left_boundary='beam', right_boundary='reflecting', source=5, left_strength=10)
    slab.create_region(material_type='thicc', length=length, num_cells=num_cells, x_left=0, total_xs=100, scatter_xs=99.5)

    perform_transport_in(slab)

    plot_scalar_flux(slab)
    plot_current(slab)


def reeds_problem():
    length = 8
    l1 = 2
    l2 = 2
    l3 = 1
    l4 = 1
    l5 = 2
    num_cells = 100
    num_angles = 2
    incident_flux = 10                  # Approximate by having mu closest to 1 have the incident flux

    slab = Slab(num_angles=num_angles, left_boundary='vacuum', right_boundary='reflecting', source=5)
    slab.create_region(material_type='thicc', length=length, num_cells=num_cells, x_left=0, total_xs=100, scatter_xs=99.5)

    perform_transport_in(slab)

    plot_scalar_flux(slab)
    plot_current(slab)

if __name__ == "__main__":
    optically_thick_diffusive_problem()
