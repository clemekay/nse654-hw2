import numpy as np
import h5py


class SimpleMaterial(object):
    def __init__(self, material_type, total_xs, scatter_xs):
        self.name = material_type
        self.total_xs = total_xs
        self.scatter_xs = scatter_xs
