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

    def init_values_grid(self, grid, t0, vx0, vy0, p0):
        '''Init values on grid for t=0'''    
        self.vx0 = None #TODO store as arrays
        self.vy0 = None
        self.p0 = None
        self.t0 = None
        

class InputAndResults():
    '''
    Class to contain inputs and results. Acts like an I/O-module.
    '''
    def __init__(self, t_final, dt_res, dt, grid,):
        '''
        t_final: final time (when to terminate simulation)
        dt_res: time interval to save to result
        dt: time interval for each iteration
        grid: class holding the grid
        '''
        self.t_final = t_final
        self.dt_res = dt_res
        self.dt = dt #TODO add more as we go on
        self.t_res = [] #TODO store in a better way using pandas?
        self.vx_res = []
        self.vy_res = []
        self.p_res = []
    
def solver_driver(m, io_res: InputAndResults, BC: BoundaryConditions):
    '''
    Driver program to take timestep and save results to array
    Input:
        m: max number of times to store in result/outer timesteps to take
        io_res: class holding Input and results
    '''
    io_res.t_res.append(BC.t0)
    io_res.vx_res.append(np.copy(BC.vx0))
    io_res.vy_res.append(np.copy(BC.vy0))
    io_res.p_res.append(np.copy(BC.p0))

    t = BC.t0
    vx, vy, p = BC.vx0, BC.vy0, BC.p0

    for _ in range(m):
        t_end = t + io_res.dt_res
        if t_end > io_res.t_final:
            t_end = io_res.t_final
        dt = io_res.dt
        solver_integrator(t, vx, vy, p, dt, t_end)

        io_res.t_res.append(t)
        io_res.vx_res.append(np.copy(vx))
        io_res.vy_res.append(np.copy(vy))
        io_res.p_res.append(np.copy(p))

        if t >= io_res.t_final: 
            break


def solver_integrator(t, vx, vy, p, dt, t_end):
    '''
    Take one output step
    '''
    
    return 0

def solver_propagator():
    '''
    Take one single timestep
    '''

    return 0

def solver_derivatives():
    '''
    Determine derivatives
    '''

    return None