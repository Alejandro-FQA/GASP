# # 1D Quantum Harmonic Oscillator

# %% import PyTorch
import torch

# impor numeric libraries
import numpy as np
import matplotlib.pyplot as plt

# import Custom Modules
import parameters as pm
import utilities as utils
import plots
# import Customs Classes
from models import Gaussian, NQS
from analysis import Dynamics
# improt Custom Functions
from integrators import integrator

#%% Default type
torch.set_default_dtype(torch.float64)
#%% General parameters --------------------------------------------------

# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)

# Seed of the random number generator
seed = 1                                       
torch.manual_seed(seed)

# Create the NN model
num_params = 1 # only 1 or 2 parameters
model = Gaussian(num_params).to(device) 

# Define Parameters
new_params = torch.view_as_complex(torch.randn(num_params,2))
# ground state values provided
new_params[0] = 1 + 0 * 1j
if num_params == 2:
    new_params[1] = 0 + 0 * 1j

# Update model parameteres
model.update_params(new_params)

# Create a spacial grid object
grid = utils.PointGrid(100, start=-8, end=8)
                                    
# device tells where to store the tensor
mesh = grid.get_points().to(device)

# torch.unsqueeze(1) creates a [Nx, 1] tensor
# torch.requires_grad_() tells autograd to record operations on this tensor
x_grid = mesh.clone().unsqueeze(1).requires_grad_().type(torch.complex128)

#%% Stochastic Reconfiguration
# Initial conditions
pm.x0 = 1
pm.w = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 5

# Integrator parameters
pm.evolution = 'imag'

# Perform imag time evolution
imag_file_path = 'model_states_imag_evo.h5'
integrator(model, x_grid, file_path=imag_file_path)

# Get the dynamics
imag_evo = Dynamics(file_path=imag_file_path, x_grid=x_grid)
# Compute density
den = np.abs(imag_evo.psi)**2
# Get parameters
params = imag_evo.get_params()

# Plot data
getattr(plots, f'evo_fig_{num_params}_params')(imag_evo.t_grid, mesh, den.T, params, fig_name='evo_imag.png')

#%% Stochastic Reconfiguration
# Initial conditions
pm.x0 = 0
pm.w = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 50

# Integrator parameters
pm.evolution = 'real'

# Perform real time evolution
real_file_path = 'model_states_real_evo.h5'
integrator(model, x_grid, file_path=real_file_path)

# Get the dynamics
real_evo = Dynamics(file_path=real_file_path, x_grid=x_grid)
# Compute density
den = np.abs(real_evo.psi)**2
# Get parameters
params = real_evo.get_params()

# Plot data
getattr(plots, f'evo_fig_{num_params}_params')(real_evo.t_grid, mesh, den.T, params, fig_name='evo_real.png')

#%%