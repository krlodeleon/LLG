# Importing packages
import numpy as np
import cupy as cp
import sys, os, shutil, time

parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
package_dir = parent_dir + '/packages'
sys.path.append(package_dir)

import crydberg as rd

# Starting time recording
start_time = time.time()

# Program variables
task_id=int(sys.argv[1]) - 1 # for SLURM_ARRAY_TASK_ID
N=int(sys.argv[2])
T_low=float(sys.argv[3])
T_high=float(sys.argv[4])
n_T=int(sys.argv[5])
n_steps_burn=int(sys.argv[6]) # burn-in time
n_steps_history=int(sys.argv[7]) # steps recorded after burn-in
n_samples_thermal=int(sys.argv[8]) # thermal samples
n_samples_parallel=int(sys.argv[9]) # disorder samples computed in parallel (with thermal samples)
n_samples_internal=int(sys.argv[10]) # number of samples in one node
n_samples_external=int(sys.argv[11]) # number of samples across nodes
precession=float(sys.argv[12]) # controls precession term in the LLG
K=float(sys.argv[13]) # anisotropy
dt=float(sys.argv[14]) # length of time step
dir_sys=sys.argv[15] # directory to store system data

for arg in sys.argv: print(arg)

# Additional system parameters
B0 = 0
B0_theta = 0
B0_phi = 0

# Thermodynamic parameters and indexing scheme
T_vals = cp.linspace(T_low, T_high, n_T)
T_idcs = np.arange(len(T_vals))
sample_idcs = np.arange(n_samples_external)
sample_idcs, T_idcs = np.meshgrid(sample_idcs, T_idcs)
sample_idcs, T_idcs = sample_idcs.flatten(), T_idcs.flatten()
T_idx = T_idcs[task_id]
sample_idx = sample_idcs[task_id]
T = T_vals[T_idx]

# File tag
tag = str(T_idx) + '_' + str(sample_idx)

# Additional dynamics parameters
alpha = 1

# number of parallel samples in one MagneticSystem instance
n_samples = n_samples_parallel*n_samples_thermal # total number of samples computed in parallel in one GPU

# running simulation for each internalSample sequentially
for internalSample_idx in range(n_samples_internal):
    # generating system
    systems = rd.MagneticSystem(N, n_samples, K=K)
    systems.align_m() # randomizing initial spin config. m0
    systems.m = cp.broadcast_to(systems.m[0], (n_samples, N, 3)) # copying one m0 throughout all samples
    np.save(f'{dir_sys}/m0_T{T_idx}.npy', systems.m.get()) # saving m0

    # setting Js of thermal samples batches with one J
    J_toSet = []
    for J in systems.J[:n_samples_parallel]:
        J_toSet.append(cp.broadcast_to(J, (n_samples_thermal, N, N)))
    J_toSet = cp.array(J_toSet)
    J_toSet = J_toSet.reshape(n_samples, N, N)
    systems.J = J_toSet
    del J_toSet # deleting buffer
    np.save(f'{dir_sys}/J_T{T_idx}.npy', systems.J.get()) # saving coupling matrices

    # burning-in system
    for i in range(n_steps_burn):
        systems.LLG_1stepSIB(dt, alpha=1, precession=0, T=T)
        print(f'\rProgress: {100*i/n_steps_burn}%     ', end='')
    
    # storing data
    history = []
    for i in range(n_steps_history):
        history.append(systems.m)
        systems.LLG_1stepSIB(dt, alpha=1, precession=0, T=T)
    history = cp.array(history).reshape(n_steps_history, n_samples_parallel, n_samples_thermal, N, 3)
    np.save(f'{dir_sys}/history_T{T_idx}.npy', history.get())
    print(f'\r{T_idx + 1}/{T_vals.size} Done.          ')

# Finishing and reporting elapsed time
end_time = np.round((time.time() - start_time)/3600,2)
print(f'Simulation {tag} done. Time elapsed: {end_time} hrs.')