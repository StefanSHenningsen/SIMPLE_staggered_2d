import numpy as np

class Grid():
    '''
    Class to contain and generate the grid. Origo is at the upper left corner
    and x-direction is horizontal (columns) left to right
    and y-direction is vertical (rows) up to down
    '''
    def __init__(self, nx, ny, lx, ly):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.grid_xx, self.grid_yy = self.generate_grid()

    def generate_grid(self):
        ''' 
        Function to generate grid - is uniform here and the grid contains 
        the boundary points. TODO try to make nonuniform + give unittest
        '''
        x_val = np.linspace(0, self.lx, self.nx + 1, endpoint=True)
        y_val = np.linspace(0, self.ly, self.ny + 1, endpoint=True)
        grid_xx,grid_yy = np.meshgrid(x_val,y_val)
        return grid_xx, grid_yy


class BoundaryConditions():
    '''
    Class to contain and define boundary conditions for a square grid and
    Dirichlet boundary conditions. The pressure does not enter in the
    equations for staggered grid.
    '''
    def __init__(self, grid: Grid, vw, ve, vn, vs):
        '''
        vw is the velocity at the western boundary
        TODO make able to deal with more complex bc + pressure
        '''
        self.init_bc_arrays(grid, vw, ve, vn, vs)
        self.init_values_zero_grid(grid)

    def init_bc_arrays(self, grid, vw, ve, vn, vs):
        '''Init bc arrays'''
        self.vw = np.full(grid.ny, vw)
        self.ve = np.full(grid.ny, ve)
        self.vn = np.full(grid.nx, vn)
        self.vs = np.full(grid.nx, vs)

    def init_values_zero_grid(self, grid: Grid):
        '''Init values on grid, here all values are ero'''    
        self.vx0 = np.zeros((grid.ny, grid.nx))
        self.vy0 = np.zeros((grid.ny, grid.nx))
        self.p0 = np.zeros((grid.ny, grid.nx))
        self.t0 = 0

class SetupAndResults():
    '''
    Class to contain inputs and results. Acts like an I/O-module.
    '''
    def __init__(self, t_final, dt_res, dt, n_max_driver, n_max_integrator,
                 n_max_outer_ite, n_max_inner_ite):
        '''
        t_final: final time (when to terminate simulation)
        dt_res: time interval to save to result
        dt: time interval for each iteration
        '''
        self.t_final = t_final
        self.dt_res = dt_res
        self.dt = dt

        self.n_max_driver = n_max_driver
        self.n_max_integrator = n_max_integrator
        self.n_max_outer_ite = n_max_outer_ite
        self.n_max_inner_ite = n_max_inner_ite

        self.t_res = [] #TODO store in a better way using pandas?
        self.vx_res = []
        self.vy_res = []
        self.p_res = []

        self.t_prev_1 = [] #TODO do this in a smarter way? prev 1 is prev
        self.vx_prev_1 = []
        self.vy_prev_1 = []
        self.p_prev_1 = []
        
        self.t_prev_2 = []#prev 2 is prev prev
        self.vx_prev_2 = []
        self.vy_prev_2 = []
        self.p_prev_2 = []


    def export_results(self):
        '''
        use to export results for further analysis and visualization
        '''
        pass #TODO
    
def solver_driver(set_res: SetupAndResults, bc: BoundaryConditions):
    '''
    Driver to take "outer" timestep, t -> t + dt_res, and save results to array
    '''
    set_res.t_res.append(bc.t0)
    set_res.vx_res.append(np.copy(bc.vx0))
    set_res.vy_res.append(np.copy(bc.vy0))
    set_res.p_res.append(np.copy(bc.p0))
    
    #TODO SETUP THE FIRST TIMESTEP IN ANTOHER WAY!
    #******************************
    set_res.t_prev_2 = bc.t0
    set_res.vx_prev_2 = np.copy(bc.vx0)
    set_res.vy_prev_2 = np.copy(bc.vy0)
    set_res.p_prev_2 = np.copy(bc.p0)
    set_res.t_prev_1 = bc.t0
    set_res.vx_prev_1 = np.copy(bc.vx0)
    set_res.vy_prev_1 = np.copy(bc.vy0)
    set_res.p_prev_1 = np.copy(bc.p0)
    #*****************************
    for _ in range(set_res.n_max_driver):
        t_end = set_res.t_res[-1] + set_res.dt_res
        if t_end > set_res.t_final:
            t_end = set_res.t_final
        t, vx, vy, p = solver_integrator(t_end, set_res)
        set_res.t_res.append(t)
        set_res.vx_res.append(np.copy(vx))
        set_res.vy_res.append(np.copy(vy))
        set_res.p_res.append(np.copy(p))
        if t >= set_res.t_final:
            break


def solver_integrator(t_end, set_res: SetupAndResults):
    '''
    Take one large timestep, t -> t + dt_res.
    '''
    t = set_res.t_res[-1]
    dt = set_res.dt
    for _ in range(set_res.n_max_integrator):
        if t_end - t < dt:
            dt = t_end - t
        t, vx, vy, p = solver_propagator(t, dt, set_res)
        #TODO UPDATE IN A BETTER WAY
        set_res.t_prev_2 = set_res.t_prev_1
        set_res.vx_prev_2 = np.copy(set_res.vx_prev_1)
        set_res.vy_prev_2 = np.copy(set_res.vy_prev_1)
        set_res.p_prev_2 = np.copy(set_res.p_prev_1)
        set_res.t_prev_1 = t
        set_res.vx_prev_1 = np.copy(vx)
        set_res.vy_prev_1 = np.copy(vy)
        set_res.p_prev_1 = np.copy(p)
        if t >= t_end:
            return t, vx, vy, p
    raise RuntimeError("n_integrator was reached before it finished")

def solver_propagator(t, dt, set_res: SetupAndResults):
    '''
    Take one single timestep going through outer iterations and 
    checking for convergence (p. 188 - ...).
    '''
    vx_ite = np.copy(set_res.vx_prev_1)
    vy_ite = np.copy(set_res.vy_prev_1)
    p_ite = np.copy(set_res.p_prev_1)
    for _ in range(set_res.n_max_outer_ite):
        vx_ite_new, vy_ite_new, p_ite_new = solver_outer_iteration(t, dt, vx_ite,
                                                       vy_ite, p_ite, set_res)
        if check_convergence_outer_ite(vx_ite_new, vy_ite_new, p_ite_new,
                                       vx_ite, vy_ite, p_ite):
            break
        vx_ite = np.copy(vx_ite_new)
        vy_ite = np.copy(vy_ite_new)
        p_ite = np.copy(p_ite_new)

    return t + dt, vx_ite_new, vy_ite_new, p_ite_new

def solver_outer_iteration(t, dt, vx_ite, vy_ite, p_ite,
                           set_res: SetupAndResults):
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
    AE, AN, AW, AS, AP, QP = get_equations_momentum(t, dt, vx_ite, vy_ite,
                                                    p_ite, set_res)
    #solve inner iteration for u
    solver_inner_iteration()
    #setup eq. for pressure
    get_equations_pressure()
    #solve inner iteration
    solver_inner_iteration()
    #correct velocity and pressure
    correct_presure_velocity()
    
    vx_ite_new, vy_ite_new, p_ite_new = vx_ite, vy_ite, p_ite

    return  vx_ite_new, vy_ite_new, p_ite_new


def solver_inner_iteration():
    '''
    make iterative solution (p. 100)
    '''
    return None

def get_equations_momentum(t, dt, vx_ite, vy_ite, p_ite,
                           set_res: SetupAndResults):
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

def check_convergence_outer_ite(vx, vy, p, vx_prev, vy_prev, p_prev):
    '''Test convergence...'''
    #TODO make correct...
    return True



#**********************************************************************
def add(x,y):
    return x + y

def divide(x,y):
    if y == 0:
        raise ValueError('Can not divide by zero!')
    return x/y

#**********************************************************************
if __name__ == "__main__":
    grid_test = Grid(3, 2, 3, 4)
    print(grid_test.grid_xx)
    print(grid_test.grid_yy)
    bc_test = BoundaryConditions(grid_test, 1,2,3,4)
    #print(bc_test.vw, bc_test.ve, bc_test.vn, bc_test.vs)
    #print(bc_test.p0, bc_test.t0, bc_test.vx0, bc_test.vy0)
    set_res_test = SetupAndResults(5, 2, 1, 10, 10, 10, 10)
    solver_driver(set_res_test, bc_test)
    print(set_res_test.t_res)
    print(set_res_test.vx_res)