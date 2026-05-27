########################
# rydberg.py ver 3.0.0 #
########################

# Includes parallel internal samples

import numpy as np
from scipy.sparse import block_diag

##### for creating a network #####
uniform_to_theta = lambda t : np.arccos(1 - 2*t)

class Network:
    def __init__(self, N, n_networks=1):
        self.N = N
        self.n_networks = n_networks
        self.J_width = 1/np.sqrt(self.N)
        
        J_out = np.triu(np.random.normal(0, self.J_width, size=(self.n_networks, self.N, self.N)), 1)
        J_out = J_out + J_out.transpose(0, 2, 1)
        self.J = J_out
        
##### magnetic system attributes and methods #####

def B_thermal(m, T, dt):
    return np.random.normal(0, np.sqrt(2*T/dt), size=np.shape(m))

def annealing_schedule(power, T_max, size):
    return np.sort(T_max*(1-np.random.power(power + 1, size)))[::-1]

class MagneticSystem(Network):
    def __init__(self, N, n_networks, B0=0, B0_theta=0, B0_phi=0, K=0):
        super().__init__(N, n_networks)
        self.B0 = B0*np.array([np.sin(B0_theta)*np.cos(B0_phi),\
                           np.sin(B0_theta)*np.sin(B0_phi),\
                           np.cos(B0_theta)]) # Setting external field
        self.K = K
        
        # Initializing state
        self.m = np.zeros((self.n_networks,self.N,3))
        self.m[:,:,2] = 1     
    
    def align_m(self, theta=None, phi=None):
        if theta == None:
            thetas = uniform_to_theta(np.random.rand(self.n_networks, self.N))
        else:
            thetas = theta*np.ones((self.n_networks, self.N))
        if phi == None:
            phis = 2*np.pi*np.random.rand(self.n_networks, self.N)
        else:
            phis = phi*np.ones((self.n_networks, self.N))
        
        mxs = np.sin(thetas)*np.cos(phis)
        mys = np.sin(thetas)*np.sin(phis)
        mzs = np.cos(thetas)
        
        self.m = np.stack((mxs, mys, mzs), axis=-1)
    
    @property
    def energy_exchange(self):
        return (1/2)*np.sum(self.m * (self.J @ self.m), axis=(1,2))
    
    @property
    def energy_Zeeman(self):
        return -np.sum(np.dot(self.m, self.B0), axis=-1)
    
    @property
    def energy_anisotropy(self):
        'Anisotropy defaulted to be along z.'
        return -self.K*np.sum(self.m[:,:,2]**2, axis=1)/2
    
    @property
    def energy(self):
        return self.energy_exchange + \
               self.energy_Zeeman + \
               self.energy_anisotropy
    
    @property
    def spinstate(self):
        return np.sign(self.m[:,:,2])

    @property
    def B(self):
        B = -self.J @ self.m # adding spin-spin interaction contributions
        B[:,:,2] += self.K*self.m[:,:,2] # adding anisotropy contributions
        B += self.B0 # adding Zeeman contributions        
        return B

    def get_dmOVERdt(self, alpha, precession, B_noise=0):
        B = self.B + B_noise

        LLG_precessional = np.cross(self.m, B)
        LLG_damping = -np.cross(self.m, LLG_precessional)
        dmOVERdt = -precession*LLG_precessional + alpha*LLG_damping
        return dmOVERdt
    
    def get_dmField(self, dt, alpha, precession, B_noise=0): # when crossed to the right of the spin, this gives the (Euler) dm for the next step. For now, applicable to overdamped case only.
        B = self.B + B_noise
        return -dt*(precession*B + alpha*np.cross(self.m, B))

    def get_SIBOperator(self, dt, alpha, dmField):
        blocks = []
        for L in dmField.reshape((self.N*self.n_networks, 3)):
            block = np.array([[ 1     , -L[2]/2,  L[1]/2],
                              [ L[2]/2,  1     , -L[0]/2],
                              [-L[1]/2,  L[0]/2,  1     ]])
            blocks.append(block)
        operator = block_diag(blocks, format='csc')
        return operator.toarray()
    
    def get_SIBBias(self, dt, alpha, dmField):
        return (self.m + np.cross(self.m, dmField)/2).flatten()
    
    def LLG_1stepSIB(self, dt, alpha, precession, T=0):
        B_noise = B_thermal(self.m, T, dt)
        
        m0 = np.copy(self.m)
        dmField = self.get_dmField(dt, alpha, precession, B_noise)
        m_SIB1 = np.linalg.solve(self.get_SIBOperator(dt, alpha, dmField), self.get_SIBBias(dt, alpha, dmField)).reshape((self.n_networks, self.N, 3)) 
        
        self.m = (m0 + m_SIB1)/2
        dmField = np.copy(self.get_dmField(dt, alpha, precession, B_noise))
        self.m = m0
        
        m_SIB2 = np.linalg.solve(self.get_SIBOperator(dt, alpha, dmField), self.get_SIBBias(dt, alpha, dmField)).reshape((self.n_networks, self.N, 3))
        self.m = m_SIB2
        
    def LLG_1stepEuler(self, dt, alpha, precession, B_noise=0):            
        m_out = self.m + dt*self.get_dmOVERdt(alpha, precession, B_noise)
        
        # Imposing unity of vector magnitudes
        norms = np.linalg.norm(m_out, axis=2)
        norms = np.broadcast_to(norms[:, :, None], (self.n_networks, self.N, 3))
        m_out = m_out/norms
        self.m = m_out
    
    def LLG_1stepHeun(self, dt, alpha, precession, T=0):
        B_noise = B_thermal(self.m, T, dt)
        
        m0 = np.copy(self.m)
        dmOVERdt_1 = np.copy(self.get_dmOVERdt(alpha, precession, B_noise))
        self.LLG_1stepEuler(dt, alpha, B_noise)
        dmOVERdt_2 = np.copy(self.get_dmOVERdt(alpha, precession, B_noise))
        m_out = m0 + dt*(dmOVERdt_1 + dmOVERdt_2)/2
        
        # Imposing unity of vector magnitudes
        norms = np.linalg.norm(m_out, axis=2)
        norms = np.broadcast_to(norms[:, :, None], (self.n_networks, self.N, 3))
        m_out = m_out/norms
        self.m = m_out
        
    
    def LLG_evolve(self, dt, steps, alpha, precession, T=0, method='LLG_1stepHeun'):
        LLG_1step = getattr(self, method)        
            
        if isinstance(T, int) or isinstance(T, float) == True:
            for i in range(steps):
                LLG_1step(dt, alpha, precession, T)
        
        elif isinstance(T, np.ndarray) == True:
            for temp in T:
                LLG_1step(dt, alpha, precession, temp)
                
        else:
            raise Exception('Unsupported data type for temperature.')