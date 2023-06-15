import matplotlib.pyplot as plt
from twod_modules.geometry import TwoDSlab
from twod_modules.transport import transport as perform_transport_in


def uniform_infinite_medium():
    Q = 8
    sigma_a = 2
    c = 0.5
    sigma_t = sigma_a / (1-c)
    incident_source = Q / sigma_a / 4
    slab = TwoDSlab(quadrature=4, left_bc='isotropic', right_bc='isotropic', top_bc='isotropic', bottom_bc='isotropic',
                    left_strength=incident_source,
                    right_strength=incident_source,
                    bottom_strength=incident_source,
                    top_strength=incident_source)
    slab.create_region(x=(0., 1.), y=(0., 1.), material_type='infinite', total_xs=sigma_t, scatter_xs=c*sigma_t, source=Q)

    slab.create_mesh(num_xcells=20, num_ycells=20)
    perform_transport_in(slab)

    slab.plot_scalar_flux()

    slab.plot_slice_scalar_flux()
    phi = Q / sigma_a
    plt.ylim(phi-.0001,phi+.0001)
    plt.show()
    print('Finished')


def whering_test_prob_four(mesh):
    slab = TwoDSlab(quadrature=8, left_bc='reflecting', right_bc='vacuum', top_bc='reflecting', bottom_bc='vacuum')
    slab.create_region(x=(0.,2.), y=(0.,2.), material_type='A', total_xs=2.0, scatter_xs=0.9*2.0, source=1.0)
    slab.create_region(x=(4.,6.), y=(0.,8.), material_type='C_bottom', total_xs=2.0, scatter_xs=0., source=0.)
    slab.create_region(x=(8.,10.), y=(4.,12.), material_type='C_bottom', total_xs=2.0, scatter_xs=0., source=0.)
    slab.create_region(x=(0.,12.), y=(0.,12.), material_type='B', total_xs=2.0, scatter_xs=0.0, source=0.0, fill=True)

    slab.create_mesh(num_xcells=mesh, num_ycells=mesh)
    perform_transport_in(slab)
    slab.plot_slice_scalar_flux()
    plt.semilogy()
    plt.show()


def source_free_half_space():
    slab = TwoDSlab(quadrature=4, left_bc='isotropic', right_bc='isotropic', top_bc='vacuum', bottom_bc='vacuum',
                    left_strength=10,
                    right_strength=6)
    slab.create_region(x=(0.,8), y=(0.,1.), material_type='A', total_xs=3, scatter_xs=1.5, source=0)
    slab.create_region(x=(8.,20.), y=(0.,1.), material_type='B', total_xs=1, scatter_xs=0.9, source=0)

    slab.create_mesh(num_xcells=100, num_ycells=100)

    perform_transport_in(slab)

    slab.plot_slice_scalar_flux()
    plt.show()



if __name__ == "__main__":
    uniform_infinite_medium()
