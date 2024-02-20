import numpy as np

class Grid():
    '''
    Class to contain and generate the grid. Origo is at the lower left corner
    and x-direction is horizontal (columns)
    '''
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.grid = self.generate_grid()

    def generate_grid(self):
        ''' Function to generate grid can be uniform or not...'''
        return 0


class BoundaryConditions():
    '''
    Cclass to contain and define boundary conditions for a square grid and
    Dirichlet boundary conditions. The pressure does not enter in the
    equations for staggered grid.
    '''
    def __init__(self, grid, v_w, v_e, v_n, v_s):
        '''v_w is the velocity at the western boundary'''
        self.bc_w = None #TODO store as arrays
        self.bc_e = None
        self.bc_n = None
        self.bc_s = None


class InputAndResults():
    '''
    Class to contain inputs and results. Acts like an I/O-module.
    '''
    def __init__(self, tf, dt_res, dt):
        '''
        tf: final time (when to terminate simulation)
        dt_res: time interval to save to result
        dt: time interval for each iteration
        '''
        self.tf = tf
        self.dt_res = dt_res
        self.dt = dt #TODO add more as we go on

