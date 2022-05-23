from modules.geometry import Slab, Region
from modules.transport import transport as perform_transport_in
from modules.plot_results import do_results_things


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
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='reflecting', right_boundary='reflecting')
    slab.create_region(material_type='infinite uniform', length=length, num_cells=num_cells, x_left=0, total_xs=1,
                       scatter_xs=0, source=4)

    perform_transport_in(slab)

    do_results_things(slab)


def source_free_pure_absorber():
    length = 10
    l1 = 3
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='isotropic', right_boundary='reflecting', left_strength=10)
    slab.create_region(material_type='infinite uniform', length=l1, num_cells=30, x_left=0, total_xs=1, scatter_xs=0, source=0)
    slab.create_region(material_type='infinite heterog', length=length - l1, num_cells=70, x_left=l1, total_xs=1,
                       scatter_xs=0, source=0)

    perform_transport_in(slab)

    do_results_things(slab)


def source_free_half_space():
    length = 20
    l1 = 8
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='isotropic', right_boundary='isotropic',
                left_strength=10, right_strength=6)
    slab.create_region(material_type='left', length=l1, num_cells=80, x_left=0, total_xs=3, scatter_xs=1.5, source=0)
    slab.create_region(material_type='right', length=length - l1, num_cells=120, x_left=l1, total_xs=1, scatter_xs=0.9, source=0)

    perform_transport_in(slab)

    do_results_things(slab)


def optically_thick_diffusive_problem():
    length = 10
    num_cells = 100
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='beam', right_boundary='reflecting', left_strength=10)
    slab.create_region(material_type='thicc', length=length, num_cells=num_cells, x_left=0, total_xs=100, scatter_xs=99.5, source=5)

    perform_transport_in(slab)

    do_results_things(slab)


def reeds_problem():
    l1 = 2
    l2 = 2
    l3 = 1
    l4 = 1
    l5 = 2
    dx = 0.1
    num_angles = 4

    slab = Slab(num_angles=num_angles, left_boundary='vacuum', right_boundary='reflecting')
    slab.create_region(material_type='r1', length=l1, num_cells=l1/dx, x_left=0, total_xs=1, scatter_xs=0.9, source=0)
    slab.create_region(material_type='r2', length=l2, num_cells=l2/dx, x_left=l1, total_xs=1, scatter_xs=0.9, source=1)
    slab.create_region(material_type='r3', length=l3, num_cells=l3/dx, x_left=l2, total_xs=0, scatter_xs=0.0, source=0)
    slab.create_region(material_type='r4', length=l4, num_cells=l4/dx, x_left=l3, total_xs=5, scatter_xs=0.0, source=0)
    slab.create_region(material_type='r5', length=l5, num_cells=l5/dx, x_left=l4, total_xs=50, scatter_xs=0.0, source=50)

    perform_transport_in(slab)
    print(slab.num_cells)
    do_results_things(slab)

if __name__ == "__main__":
    source_free_half_space()
