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
    def __init__(self, t_final, dt_res, dt, grid):
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
    
def solver_driver(n_driver,n_integrator, io_res: InputAndResults, bc: BoundaryConditions):
    '''
    Driver program to take timestep and save results to array
    Input:
        n_driver: max number of times to store in result/outer timesteps to take
        n_integrator: max number of timesteps in integrator
        io_res: class holding Input and results
    '''
    io_res.t_res.append(bc.t0)
    io_res.vx_res.append(np.copy(bc.vx0))
    io_res.vy_res.append(np.copy(bc.vy0))
    io_res.p_res.append(np.copy(bc.p0))

    t = bc.t0
    vx, vy, p = bc.vx0, bc.vy0, bc.p0

    for _ in range(n_driver):
        t_end = t + io_res.dt_res
        if t_end > io_res.t_final:
            t_end = io_res.t_final
        dt = io_res.dt
        solver_integrator(t, vx, vy, p, dt, t_end, n_integrator)

        io_res.t_res.append(t)
        io_res.vx_res.append(np.copy(vx))
        io_res.vy_res.append(np.copy(vy))
        io_res.p_res.append(np.copy(p))

        if t >= io_res.t_final:
            break


def solver_integrator(t, vx, vy, p, dt, t_end, n_integrator):
    '''
    Take one output step
    '''
    for _ in range(n_integrator):
        if (t_end - t < dt): 
            dt = t_end - t
        solver_propagator(t, vx, vy, p, dt)
        if t > t_end:
            return None
    raise RuntimeError("n_integrator was reached before it finished")

def solver_propagator(t, vx, vy, p, dt):
    '''
    Take one single timestep p. 188
    '''
    #loop in outer iterations
    solver_outer_iteration(t, vx, vy, p, dt)
    #go back and do it again + check for convergence




    return 0

def solver_outer_iteration(t, vx, vy, p, dt):
    '''
    make sequential under-relaxation (p. 118)
    setup p.178 and p. 188
    1. use last un and pn as starting point for un+1, pn+1
    2. solve linearized eq. for momentum to get um*
    3. solve pressure eq. to get p'
    4. correct velocity um to satisfy cont. and the new pressure pm
    5. go back to 2. and do it all again (outer iteration)
    6. when converged to go next timestep
    '''
    #setup eq. for momentum
    AE, AN, AW, AS, AP, QP = get_equations_momentum(vx, vy, p, dt)
    #solve inner iteration for u
    solver_inner_iteration()
    #setup eq. for pressure
    get_equations_pressure()
    #sove inner iteration
    solver_inner_iteration()
    #correct velocity and pressure
    correct_presure_velocity()
    return None


def solver_inner_iteration():
    '''
    make iterative solution (p. 100)
    '''
    return None


def get_equations_momentum(vx, vy, p, dt):
    '''
    make system of equations to solve
    Ap*up + AL*ul = Qp (eq. 7.97) 
    '''
    return None, None, None, None, None, None

def get_equations_pressure():
    '''
    make system of equations to solve
    Ap*p'p + AL*p'l = dm (eq. 7.111) 
    '''
    return None

def correct_presure_velocity():
    '''
    eq. 7.107??
    '''
    return None