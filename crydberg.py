#############################
# crydberg.py version 3.0.0 #
#############################

# Includes semi-implicit integration methods

##### Importing packages #####
from numba import cuda
import cupy as cp

##### CUDA variables and indexing #####
device = cuda.get_current_device()
maxThreadsPerBlock = device.MAX_THREADS_PER_BLOCK

def get_CUDA_block_thread_count(n_iter):
    global maxThreadsPerBlock
    if n_iter <= maxThreadsPerBlock:
        n_blocks, n_threads = 1, n_iter
    else:
        n_blocks, n_threads = n_iter//maxThreadsPerBlock + 1, maxThreadsPerBlock
    return n_blocks, n_threads    
        
##### CUDA functions #####
@cuda.jit(device=True)
def forward_elimination(M, rhs):
    for k in range(3):
        pivot = M[k, k]
        for j in range(k, 3):
            M[k, j] /= pivot
        rhs[k] /= pivot

        for i in range(k+1, 3):
            factor = M[i, k]
            for j in range(k, 3):
                M[i, j] -= factor * M[k, j]
            rhs[i] -= factor * rhs[k]

@cuda.jit(device=True)
def back_substitution(M, rhs, x):
    for i in range(2, -1, -1):
        s = rhs[i]
        for j in range(i+1, 3):
            s -= M[i, j] * x[j]
        x[i] = s

@cuda.jit(device=True)
def solve3x3(A, b, x):
    # Copy into local arrays
    M = cuda.local.array((3, 3), dtype=cp.float32)
    rhs = cuda.local.array(3, dtype=cp.float32)
    
    for i in range(3):
        rhs[i] = b[i]
        for j in range(3):
            M[i, j] = A[i, j]

    forward_elimination(M, rhs)
    back_substitution(M, rhs, x)

@cuda.jit
def kernel(A, b, x, storage):
    i = cuda.grid(1)
    solve3x3(A, b, x)
    for j in range(3):
        storage[i,j] = x[j]
        
@cuda.jit
def _solve_SIB(m, SIBOperator, SIBBias, m_out):
    i = cuda.grid(1)
    if i < len(m):
        # Defining local arrays
        m0 = cuda.local.array(3, dtype=cp.float32)
        operator = cuda.local.array((3,3), dtype=cp.float32)
        bias = cuda.local.array(3, dtype=cp.float32)
        m_next = cuda.local.array(3, dtype=cp.float32)
        
        for j, val in enumerate(m[i]): m0[j] = val
        for j, val in enumerate(SIBBias[i]): bias[j] = val
        for j in range(3):
            for k in range(3):
                operator[j,k] = SIBOperator[i,j,k]
        
        solve3x3(operator, bias, m_next)
        for j, val in enumerate(m_next): m_out[i,j] = val
    

##### Network creation #####
uniform_to_theta = lambda t : cp.arccos(1 - 2*t)

class Network:
    def __init__(self, N, n_networks=1):
        self.N = N
        self.n_networks = n_networks
        
        self.J = cp.triu(cp.random.normal(0, 1/cp.sqrt(self.N), size=(self.n_networks, self.N, self.N)), 1)
        self.J = self.J + self.J.transpose(0, 2, 1)
        
##### Magnetic system #####

def B_thermal(m, T, dt):
    return cp.random.normal(0, cp.sqrt(2*T/dt), size=cp.shape(m))

def annealing_schedule(power, T_max, size):
    return cp.sort(T_max*(1-cp.random.power(power + 1, size)))[::-1]

class MagneticSystem(Network):
    def __init__(self, N, n_networks, B0=0, B0_theta=0, B0_phi=0, K=0):
        super().__init__(N, n_networks)
        self.B0 = B0*cp.array([cp.sin(B0_theta)*cp.cos(B0_phi),\
                           cp.sin(B0_theta)*cp.sin(B0_phi),\
                           cp.cos(B0_theta)]) # Setting external field
        self.K = K
        
        # Initializing state
        self.m = cp.zeros((self.n_networks,self.N,3))
        self.m[:,:,2] = 1  
        
        # CUDA
        self.n_blocks, self.n_threads = get_CUDA_block_thread_count(self.n_networks*self.N)            
    
    def align_m(self, theta=None, phi=None):
        if theta == None:
            thetas = uniform_to_theta(cp.random.rand(self.n_networks, self.N))
        else:
            thetas = theta*cp.ones((self.n_networks, self.N))
        if phi == None:
            phis = 2*cp.pi*cp.random.rand(self.n_networks, self.N)
        else:
            phis = phi*cp.ones((self.n_networks, self.N))
        
        mxs = cp.sin(thetas)*cp.cos(phis)
        mys = cp.sin(thetas)*cp.sin(phis)
        mzs = cp.cos(thetas)
        
        self.m = cp.stack((mxs, mys, mzs), axis=-1)
    
    def set_m(self, m):
        assert cp.shape(m) == (self.n_networks, self.N, 3)
        self.m = m
    
    @property
    def energy_exchange(self):
        return (1/2)*cp.sum(self.m * (self.J @ self.m), axis=(1,2))
    
    @property
    def energy_Zeeman(self):
        return -cp.sum(cp.dot(self.m, self.B0), axis=-1)
    
    @property
    def energy_anisotropy(self):
        'Anisotropy defaulted to be along z.'
        return -self.K*cp.sum(self.m[:,:,2]**2, axis=1)/2
    
    @property
    def energy(self):
        return self.energy_exchange + \
               self.energy_Zeeman + \
               self.energy_anisotropy
    
    @property
    def spinstate(self):
        return cp.sign(self.m[:,:,2])

    @property
    def B(self):
        B = -self.J @ self.m # adding spin-spin interaction contributions
        B[:,:,2] += self.K*self.m[:,:,2] # adding anisotropy contributions
        B += self.B0 # adding Zeeman contributions        
        return B

    def get_dmOVERdt(self, alpha, precession, B_noise=0):
        B = self.B + B_noise

        LLG_precessional = cp.cross(self.m, B)
        LLG_damping = -cp.cross(self.m, LLG_precessional)
        dmOVERdt = -precession*LLG_precessional + alpha*LLG_damping
        return dmOVERdt
    
    def get_dmField(self, dt, alpha, precession, B_noise=0): # when crossed to the right of the spin, this gives the (Euler) dm for the next step. For now, applicable to overdamped case only.
        B = self.B + B_noise
        return -dt*(precession*B + alpha*cp.cross(self.m, B))

    def get_SIBOperator(self, dmField):
        Lx, Ly, Lz = dmField[:,:,0], dmField[:,:,1], dmField[:,:,2]
    
        out = cp.empty((self.n_networks, self.N, 3, 3), dtype=dmField.dtype)
    
        out[:,:,0,0] = 1
        out[:,:,1,1] = 1
        out[:,:,2,2] = 1
        out[:,:,0,1] = -Lz/2
        out[:,:,0,2] =  Ly/2
        out[:,:,1,0] =  Lz/2
        out[:,:,1,2] = -Lx/2
        out[:,:,2,0] = -Ly/2
        out[:,:,2,1] =  Lx/2
    
        return out
    
    def get_SIBBias(self, dmField):
        return self.m + cp.cross(self.m, dmField)/2
    
    def solve_SIB(self, SIBOperator, SIBBias):
        m = self.m.reshape((self.N*self.n_networks, 3))
        m_out = cp.empty((self.N*self.n_networks, 3))
        SIBOperator = SIBOperator.reshape((self.N*self.n_networks, 3, 3))
        SIBBias = SIBBias.reshape((self.N*self.n_networks, 3))
        
        _solve_SIB[self.n_blocks, self.n_threads](m, SIBOperator, SIBBias, m_out)
        
        return m_out.reshape((self.n_networks, self.N, 3))
        
    def LLG_1stepSIB(self, dt, alpha, precession, T=0):
        B_noise = B_thermal(self.m, T, dt)
        
        m0 = cp.copy(self.m)
        m_SIB1 = cp.empty_like(m0)
        
        # SIB step 1
        dmField = self.get_dmField(dt, alpha, precession, B_noise)
        operator = self.get_SIBOperator(dmField)
        bias = self.get_SIBBias(dmField)
        m_SIB1 = self.solve_SIB(operator, bias)
        
        # SIB step 2
        self.m = (m0 + m_SIB1)/2
        dmField = cp.copy(self.get_dmField(dt, alpha, precession, B_noise))
        self.m = m0 # It is important that this line comes before the next two lines (new definitons of "operator" and "bias").
        operator = self.get_SIBOperator(dmField)
        bias = self.get_SIBBias(dmField)       
        self.m = self.solve_SIB(operator, bias)
        
    # def LLG_1stepHeun(self, dt, alpha, precession, T=0):
    #     B_noise = B_thermal(self.m, T, dt)
        
    #     m0 = self.m
    #     dmOVERdt_1 = self.get_dmOVERdt(alpha, precession, B_noise)
    #     self.LLG_1stepEuler(dt, alpha, B_noise)
    #     dmOVERdt_2 = self.get_dmOVERdt(alpha, precession, B_noise)
    #     m_out = m0 + dt*(dmOVERdt_1 + dmOVERdt_2)/2
        
    #     # Imposing unity of vector magnitudes
    #     norms = cp.linalg.norm(m_out, axis=1)
    #     m_out = m_out/norms[:, cp.newaxis]
        
    #     self.m = m_out
    
    def LLG_evolve(self, dt, steps, alpha, precession, T=0, method='LLG_1stepHeun'):
        LLG_1step = getattr(self, method)        
            
        if T.size == 1:
            for i in range(steps):
                LLG_1step(dt, alpha, precession, T)
        
        elif T.size != 1:
            for temp in T:
                LLG_1step(dt, alpha, precession, temp)
                
        else:
            raise Exception('Unsupported data type for temperature.')
    
