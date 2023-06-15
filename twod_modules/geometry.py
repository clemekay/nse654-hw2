from twod_modules import level_symmetric_quadrature
import numpy as np
import matplotlib.pyplot as plt


def boundary_condition_validity(boundary_condition):
    if boundary_condition == 'reflecting' or boundary_condition == 'vacuum' or boundary_condition == 'beam' or boundary_condition == 'isotropic':
        return True
    else:
        return False


class TwoDSlab:
    def __init__(self, quadrature, left_bc, right_bc, top_bc, bottom_bc, left_strength=None, right_strength=None,
                 top_strength=None, bottom_strength=None):
        self.tolerance = 1e-6
        # Spatial definitions
        self.region = []
        self.x_boundary = np.array([])
        self.y_boundary = np.array([])
        self.x = None
        self.y = None
        self.length = None
        self.height = None
        self.dx = None
        self.dy = None
        self.num_xcells = 0
        self.num_ycells = 0
        # Angular parameters
        self.num_octant_ordinates = int(quadrature * (quadrature + 2) / 8)  # Per-octant
        self.mu, self.eta, self.weight = level_symmetric_quadrature.define_ordinates(quadrature)
        # Cell-wise information
        self.fixed_source = None
        self.scalar_flux = None
        self.current = None
        self.total_xs = None
        self.scatter_xs = None
        self.fill = None
        # Boundary definitions
        self.left_boundary_condition = left_bc
        self.right_boundary_condition = right_bc
        self.bottom_boundary_condition = bottom_bc
        self.top_boundary_condition = top_bc
        self.left_incident_strength = left_strength
        self.right_incident_strength = right_strength
        self.bottom_incident_strength = bottom_strength
        self.top_incident_strength = top_strength
        self.left_boundary = None
        self.right_boundary = None
        self.top_boundary = None
        self.bottom_boundary = None

        assert boundary_condition_validity(self.right_boundary_condition) is True, "Unreadable right boundary condition"
        assert boundary_condition_validity(self.bottom_boundary_condition) is True, "Unreadable left boundary condition"
        assert boundary_condition_validity(self.top_boundary_condition) is True, "Unreadable right boundary condition"

    def create_region(self, x: tuple[float, float], y: tuple[float, float],
                      material_type, total_xs, scatter_xs, source, fill=False):
        assert x[1] - x[0] > 0, "Region length must be > 0"
        assert y[1] - y[0] > 0, "Region height must be > 0"
        self.x_boundary = np.unique(np.append(self.x_boundary, x))
        self.y_boundary = np.unique(np.append(self.y_boundary, y))
        if fill:
            assert self.fill is None, "Only one fill material may be defined"
            self.fill = Region(x, y, material_type, total_xs, scatter_xs, source)
        else:
            self.region.append(Region(x, y, material_type, total_xs, scatter_xs, source))

    def create_mesh(self, num_xcells, num_ycells):
        def truncate_final_cell():
            all_but_final_cell = np.sum(self.dx[0:self.num_xcells - 1])
            self.dx[self.num_xcells - 1] = self.length - all_but_final_cell

        def ensure_cell_single_material():
            for region in self.region:
                for x in region.x:
                    assert (x in self.x), "X-axis mesh creates cell with multiple materials."
                for y in region.y:
                    assert (y in self.y), "Y-axis mesh creates cell with multiple materials."
                # To-do:
                # if x not in self.x:
                # Find what cell that boundary falls into
                # Split the cell at the boundary
                # Ensure that the new mesh location does not fall into a region

        # Define outer limits of problem
        self.x_boundary.sort()
        self.y_boundary.sort()
        self.length = self.x_boundary[-1] - self.x_boundary[0]
        self.height = self.y_boundary[-1] - self.y_boundary[0]
        # Create mesh using number of cells
        # Assume uniform mesh for now
        self.x = np.linspace(self.x_boundary[0], self.x_boundary[-1], num_xcells + 1)
        self.y = np.linspace(self.y_boundary[0], self.y_boundary[-1], num_ycells + 1)
        ensure_cell_single_material()
        self.num_xcells = num_xcells
        self.num_ycells = num_ycells
        self.dx = self.x[1]
        self.dy = self.y[1]
        # Create boundary conditions
        # First index is quadrature direction (++, -+, --, +-), second is node
        self.left_boundary = np.zeros((4, 2, self.num_octant_ordinates, num_ycells))
        self.right_boundary = self.left_boundary.copy()
        self.top_boundary = np.zeros((4, 2, self.num_octant_ordinates, num_xcells))
        self.bottom_boundary = self.top_boundary.copy()
        # Create solution arrays
        self.scalar_flux = np.zeros((4, num_xcells, num_ycells))
        self.current = np.zeros((4, num_xcells, num_ycells))

    def create_material_data_arrays(self):
        self.fixed_source = 1000*np.ones((4, self.num_xcells, self.num_ycells))
        self.total_xs = 1000*np.ones((self.num_xcells, self.num_ycells))
        self.scatter_xs = self.total_xs.copy()

        x_mid = np.linspace(self.x_boundary[0] + self.dx/2, self.x_boundary[-1] - self.dx/2, self.num_xcells)
        y_mid = np.linspace(self.y_boundary[0] + self.dy/2, self.y_boundary[-1] - self.dy/2, self.num_ycells)

        for region in self.region:
            which_xcells = np.searchsorted(x_mid, region.x)
            which_ycells = np.searchsorted(y_mid, region.y)
            self.total_xs[which_xcells[0]:which_xcells[1], which_ycells[0]:which_ycells[1]] = region.total_xs
            self.scatter_xs[which_xcells[0]:which_xcells[1], which_ycells[0]:which_ycells[1]] = region.scatter_xs
            self.fixed_source[:, which_xcells[0]:which_xcells[1], which_ycells[0]:which_ycells[1]] = region.source

        if self.fill is not None:
            self.total_xs[self.total_xs == 1000] = self.fill.total_xs
            self.scatter_xs[self.scatter_xs == 1000] = self.fill.scatter_xs
            self.fixed_source[self.fixed_source == 1000] = self.fill.source

        assert (self.total_xs != 1000).all(), "Not all total_xs were set."
        assert (self.scatter_xs != 1000).all(), "Not all scatter_xs were set."

    def scattering_source_contribution(self):
        scattering_source = self.scatter_xs * self.scalar_flux
        return scattering_source

    def perform_angular_flux_sweep(self, ordinate, mu_dir, eta_dir, Q_k):
        def implement_left_boundary_condition(angle):
            # [4, 2, self.num_ordinates, num_ycells]. First index is sign of ordinates
            # Second index is node (0 is node 1, 1 is node 4)
            if self.left_boundary_condition == 'reflecting':
                self.left_boundary[0, :, angle, :] = self.left_boundary[1, :, angle, :]
                self.left_boundary[3, :, angle, :] = self.left_boundary[2, :, angle, :]
            elif self.left_boundary_condition == 'vacuum':
                self.left_boundary[:] = 0
            elif self.left_boundary_condition == 'isotropic':
                self.left_boundary[0] = self.left_incident_strength
                self.left_boundary[3] = self.left_incident_strength

        def implement_right_boundary_condition(angle):
            # [4, 2, self.num_ordinates, num_ycells]. First index is sign of ordinates
            # Second index is node (0 is node 2, 1 is node 3)
            if self.right_boundary_condition == 'reflecting':
                self.right_boundary[1, :, angle, :] = self.right_boundary[0, :, angle, :]
                self.right_boundary[2, :, angle, :] = self.right_boundary[3, :, angle, :]
            elif self.right_boundary_condition == 'vacuum':
                self.right_boundary[:] = 0
            elif self.right_boundary_condition == 'isotropic':
                self.right_boundary[1] = self.right_incident_strength
                self.right_boundary[2] = self.right_incident_strength

        def implement_bottom_boundary_condition(angle):
            # [4, 2, self.num_ordinates, num_ycells]. First index is sign of ordinates
            # Second index is node (0 is node 1, 1 is node 2)
            if self.bottom_boundary_condition == 'reflecting':
                self.bottom_boundary[0, :, angle, :] = self.bottom_boundary[3, :, angle, :]
                self.bottom_boundary[1, :, angle, :] = self.bottom_boundary[2, :, angle, :]
            elif self.bottom_boundary_condition == 'vacuum':
                self.bottom_boundary[:] = 0
            elif self.bottom_boundary_condition == 'isotropic':
                self.bottom_boundary[0] = self.bottom_incident_strength
                self.bottom_boundary[1] = self.bottom_incident_strength

        def implement_top_boundary_condition(angle):
            # [4, 2, self.num_ordinates, num_ycells]. First index is sign of ordinates
            # Second index is node (0 is node 3, 1 is node 4)
            if self.top_boundary_condition == 'reflecting':
                self.top_boundary[2, :, angle, :] = self.top_boundary[1, :, angle, :]
                self.top_boundary[3, :, angle, :] = self.top_boundary[0, :, angle, :]
            elif self.top_boundary_condition == 'vacuum':
                self.top_boundary[:] = 0
            elif self.top_boundary_condition == 'isotropic':
                self.top_boundary[2] = self.top_incident_strength
                self.top_boundary[3] = self.top_incident_strength

        def sweep_right_up():
            implement_left_boundary_condition(ordinate)
            implement_bottom_boundary_condition(ordinate)
            bottom_edge = np.array([self.bottom_boundary[0, 0, ordinate, :], self.bottom_boundary[0, 1, ordinate, :]])      # first and second node of bottom row
            psi_left_edge = np.zeros(4)
            psi_bottom_edge = np.zeros(4)
            for row in range(self.num_ycells):
                psi_left_edge[0] = self.left_boundary[0, 0, ordinate, row]
                psi_left_edge[3] = self.left_boundary[0, 1, ordinate, row]
                for col in range(self.num_xcells):
                    # Bottom boundaries for this cell
                    psi_bottom_edge[0] = bottom_edge[0, col]
                    psi_bottom_edge[1] = bottom_edge[1, col]
                    source_term = np.matmul(M, Q_k[:, col, row])
                    rhs = source_term - np.matmul(Ul, psi_left_edge) - np.matmul(Nb, psi_bottom_edge)
                    A = Lx + Ly + self.total_xs[col, row]*M
                    coeff_matrix = A + Ur + Nt
                    one_direction_angular_flux[:, col, row] = np.linalg.solve(coeff_matrix, rhs)

                    # Left edges for next cell are right edges of this cell
                    psi_left_edge[0] = one_direction_angular_flux[1, col, row]
                    psi_left_edge[3] = one_direction_angular_flux[2, col, row]
                # Bottom boundaries for next row are top boundaries of this row
                bottom_edge = np.array([one_direction_angular_flux[3, :, row], one_direction_angular_flux[2, :, row]])
            # Sweep finished, update boundaries for this angle
            self.right_boundary[0, 0, ordinate, :] = one_direction_angular_flux[1, -1, :]
            self.right_boundary[0, 1, ordinate, :] = one_direction_angular_flux[2, -1, :]
            self.top_boundary[0, 0, ordinate, :] = one_direction_angular_flux[2, :, -1]
            self.top_boundary[0, 1, ordinate, :] = one_direction_angular_flux[3, :, -1]

        def sweep_left_up():
            implement_right_boundary_condition(ordinate)
            implement_bottom_boundary_condition(ordinate)
            bottom_edge = np.array([self.bottom_boundary[1, 0, ordinate, :], self.bottom_boundary[1, 1, ordinate, :]])      # first and second node of bottom row
            psi_right_edge = np.zeros(4)
            psi_bottom_edge = np.zeros(4)
            for row in range(self.num_ycells):
                psi_right_edge[1] = self.right_boundary[1, 0, ordinate, row]
                psi_right_edge[2] = self.right_boundary[1, 1, ordinate, row]
                for col in reversed(range(self.num_xcells)):
                    # Bottom boundaries for this cell
                    psi_bottom_edge[0] = bottom_edge[0, col]
                    psi_bottom_edge[1] = bottom_edge[1, col]
                    source_term = np.matmul(M, Q_k[:, col, row])
                    rhs = source_term - np.matmul(Ur, psi_right_edge) - np.matmul(Nb, psi_bottom_edge)
                    A = Lx + Ly + self.total_xs[col, row]*M
                    coeff_matrix = A + Ul + Nt
                    one_direction_angular_flux[:, col, row] = np.linalg.solve(coeff_matrix, rhs)

                    # Right edges for next cell are left edges of this cell
                    psi_right_edge[1] = one_direction_angular_flux[0, col, row]
                    psi_right_edge[2] = one_direction_angular_flux[3, col, row]
                # Bottom boundaries for next row are top boundaries of this row
                bottom_edge = np.array([one_direction_angular_flux[3, :, row], one_direction_angular_flux[2, :, row]])
            # Sweep finished, update boundaries for this angle
            self.left_boundary[1, 0, ordinate, :] = one_direction_angular_flux[0, 0, :]
            self.left_boundary[1, 1, ordinate, :] = one_direction_angular_flux[3, 0, :]
            self.top_boundary[1, 0, ordinate, :] = one_direction_angular_flux[2, :, -1]
            self.top_boundary[1, 1, ordinate, :] = one_direction_angular_flux[3, :, -1]

        def sweep_left_down():
            implement_right_boundary_condition(ordinate)
            implement_top_boundary_condition(ordinate)
            top_edge = np.array([self.top_boundary[2, 0, ordinate, :], self.top_boundary[2, 1, ordinate, :]])      # third and fourth node of top row
            psi_right_edge = np.zeros(4)
            psi_top_edge = np.zeros(4)
            for row in reversed(range(self.num_ycells)):
                psi_right_edge[1] = self.right_boundary[1, 0, ordinate, row]
                psi_right_edge[2] = self.right_boundary[1, 1, ordinate, row]
                for col in reversed(range(self.num_xcells)):
                    # Top boundaries for this cell
                    psi_top_edge[2] = top_edge[0, col]
                    psi_top_edge[3] = top_edge[1, col]
                    source_term = np.matmul(M, Q_k[:, col, row])
                    rhs = source_term - np.matmul(Ur, psi_right_edge) - np.matmul(Nt, psi_top_edge)
                    A = Lx + Ly + self.total_xs[col, row]*M
                    coeff_matrix = A + Ul + Nb
                    one_direction_angular_flux[:, col, row] = np.linalg.solve(coeff_matrix, rhs)

                    # Right edges for next cell are left edges of this cell
                    psi_right_edge[1] = one_direction_angular_flux[0, col, row]
                    psi_right_edge[2] = one_direction_angular_flux[3, col, row]
                # Top boundaries for next row are bottom boundaries of this row
                top_edge = np.array([one_direction_angular_flux[0, :, row], one_direction_angular_flux[1, :, row]])
            # Sweep finished, update boundaries for this angle
            self.left_boundary[2, 0, ordinate, :] = one_direction_angular_flux[0, 0, :]
            self.left_boundary[2, 1, ordinate, :] = one_direction_angular_flux[3, 0, :]
            self.bottom_boundary[2, 0, ordinate, :] = one_direction_angular_flux[0, :, 0]
            self.bottom_boundary[2, 1, ordinate, :] = one_direction_angular_flux[1, :, 0]

        def sweep_right_down():
            implement_left_boundary_condition(ordinate)
            implement_top_boundary_condition(ordinate)
            top_edge = np.array([self.top_boundary[3, 0, ordinate, :], self.top_boundary[3, 1, ordinate, :]])
            psi_left_edge = np.zeros(4)
            psi_top_edge = np.zeros(4)
            for row in reversed(range(self.num_ycells)):
                psi_left_edge[0] = self.left_boundary[0, 0, ordinate, row]
                psi_left_edge[3] = self.left_boundary[0, 1, ordinate, row]
                for col in range(self.num_xcells):
                    # Top boundaries for this cell
                    psi_top_edge[2] = top_edge[0, col]
                    psi_top_edge[3] = top_edge[1, col]
                    source_term = np.matmul(M, Q_k[:, col, row])
                    rhs = source_term - np.matmul(Ul, psi_left_edge) - np.matmul(Nt, psi_top_edge)
                    A = Lx + Ly + self.total_xs[col, row]*M
                    coeff_matrix = A + Ur + Nb
                    one_direction_angular_flux[:, col, row] = np.linalg.solve(coeff_matrix, rhs)

                    # Left edges for next cell are right edges of this cell
                    psi_left_edge[0] = one_direction_angular_flux[1, col, row]
                    psi_left_edge[3] = one_direction_angular_flux[2, col, row]
                # Top boundaries for next row are bottom boundaries of this row
                top_edge = np.array([one_direction_angular_flux[1, :, row], one_direction_angular_flux[0, :, row]])
            # Sweep finished, update boundaries for this angle
            self.right_boundary[3, 0, ordinate, :] = one_direction_angular_flux[1, -1, :]
            self.right_boundary[3, 1, ordinate, :] = one_direction_angular_flux[2, -1, :]
            self.bottom_boundary[3, 0, ordinate, :] = one_direction_angular_flux[0, :, 0]
            self.bottom_boundary[3, 1, ordinate, :] = one_direction_angular_flux[1, :, 0]

        Ur = (self.mu[ordinate]*mu_dir)/self.dx*np.array([[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]]) / 6
        Ul = (self.mu[ordinate]*mu_dir)/self.dx*np.array([[-2, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, -2]]) / 6
        Nt = (self.eta[ordinate]*eta_dir)/self.dy*np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]]) / 6
        Nb = (self.eta[ordinate]*eta_dir)/self.dy*np.array([[-2, -1, 0, 0], [-1, -2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) / 6
        Lx = (self.mu[ordinate]*mu_dir)/self.dx*np.array([[2, 2, 1, 1], [-2, -2, -1, -1], [-1, -1, -2, -2], [1, 1, 2, 2]]) / 12
        Ly = (self.eta[ordinate]*eta_dir)/self.dy*np.array([[2, 1, 1, 2], [1, 2, 2, 1], [-1, -2, -2, -1], [-2, -1, -1, -2]]) / 12
        M = np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]]) / 36
        one_direction_angular_flux = np.zeros((4, self.num_xcells, self.num_ycells))        # [BLD index, column (i-index), row (j-index)]
        if mu_dir > 0 and eta_dir > 0:
            sweep_right_up()
        elif mu_dir < 0 < eta_dir:
            sweep_left_up()
        elif mu_dir < 0 and eta_dir < 0:
            sweep_left_down()
        elif mu_dir > 0 > eta_dir:
            sweep_right_down()
        return one_direction_angular_flux

    def check_scalar_flux_convergence(self, new_scalar_flux):
        if np.max(np.abs(new_scalar_flux - self.scalar_flux) / new_scalar_flux) <= self.tolerance:
            return True
        else:
            return False

    def plot_scalar_flux(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_plot = np.zeros(2*len(self.x)-2)
        y_plot = np.zeros(2*len(self.y)-2)
        flux_plot = np.zeros((len(x_plot), len(y_plot)))
        for i in range(1,len(self.x)):
            x_plot[i*2-1:i*2+1] = self.x[i]
            y_plot[i*2-1:i*2+1] = self.y[i]
        for row in range(self.num_ycells):
            for col in range(self.num_xcells):
                flux_plot[row*2, col*2:col*2+2] = self.scalar_flux[0:2, col, row]
                flux_plot[row*2+1, col*2:col*2+2] = np.flip(self.scalar_flux[2:4, col, row])

        X, Y = np.meshgrid(x_plot, y_plot)
        C = ax.contourf(X, Y, flux_plot)
        plt.title('Scalar Flux')
        plt.xlabel('x')
        plt.ylabel('y')
        fig.colorbar(C)
        plt.show()

    def plot_slice_scalar_flux(self):
        """Plot scalar flux along a horizontal cut at halfway point of y-axis. Plots at cell x-midpoint.
        Assumes mesh is uniform in x direction, ie that dx is constant. Does not require dx=dy.

        Hardcodes the fact that in the row below the y-midplane, basis function=0 for nodes 0 and 1;
        in the row above the y-midplane, basis function=0 for nodes 2 and 3.
        At a cell x-midpoint, both x-bases are 1/2.
        """

        x_mid = self.x[:-1] + self.dx/2
        half_height_index = int(self.num_ycells / 2)
        below_center_scalar_flux = np.average(self.scalar_flux[2:,:,half_height_index-1], axis=0)
        above_center_scalar_flux = np.average(self.scalar_flux[0:2,:,half_height_index], axis=0)

        plt.scatter(x_mid, below_center_scalar_flux, label='Below')
        plt.scatter(x_mid, above_center_scalar_flux, label='Above')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Scalar flux')
        plt.title('Scalar flux at y=6, 24x24 mesh')


class Region:
    def __init__(self, x: tuple[float, float], y: tuple[float, float],
                 material_type, total_xs, scatter_xs, source):
        self.material = material_type
        self.length = x[1] - x[0]
        self.height = y[1] - y[0]
        self.x = x
        self.y = y
        self.total_xs = total_xs
        self.scatter_xs = scatter_xs
        self.source = source
