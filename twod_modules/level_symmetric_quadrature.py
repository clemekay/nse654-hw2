import numpy as np
import h5py


def define_ordinates(n):
    """ Defines positive-octant ordinate coordinates for X-Y geometry.

    Level symmetric quadrature sets from [Lewis, Miller]. Available
    quadrature sets are S4, S6, S8, S12, and S16.

    Parameters
    ----------
    n : int
        SN quadrature level, where there are N(N+2)/8 ordinates per octant

    Returns
    -------
    mu : ndarray
        Ordered array of mu values for a single octant
    eta : ndarray
        Ordered array of eta values for a single octant
    w : ndarray
        Ordered array of weights for a single octant

    """
    assert n in [4, 6, 8, 12, 16], "Available quadrature sets are S4, S6, S8, S12, and S16."

    with h5py.File('twod_modules/level_symmetric_ordinates.h5', 'r') as f:
        nodes = f['S'+str(n)]['nodes'][:]
        weights = f['S'+str(n)]['weights'][:]
        point_weights = f['S'+str(n)]['weight_assignments'][:]
    reverse_ordered_nodes = np.flip(nodes)
    point_weights -= 1          # Shift index back to align with Python indexing

    n_ordinates = int(n*(n+2)/8)
    mu = np.zeros(n_ordinates)
    eta = np.zeros(n_ordinates)
    w = np.zeros(n_ordinates)

    # Defines ordinates starting at top of triangle and going row by row, left to right
    # This cannot be the best way to assign the points, but it works!
    start = 0
    chunk = 1
    row = 1

    for i in range(2, n+1):
        mu[start:chunk] = reverse_ordered_nodes[-row]
        eta[start:chunk] = nodes[0:row]
        w[start:chunk] = weights[point_weights[start:chunk]]
        start = chunk
        chunk += i

    return mu, eta, w
