# # 1D Time evolution Harmonic Oscillator

# %% import PyTorch
import torch

# impor numeric libraries
import numpy as np
import matplotlib.pyplot as plt

# import Custom Modules
import parameters as pm
import utilities as utils
# import Customs Classes
from models import Gaussian
from analysis import Dynamics
# improt Custom Functions
from integrators import integrator

#%% Default type
torch.set_default_dtype(torch.float64)
#%% General parameters --------------------------------------------------

# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)

# Create the NN model
model = Gaussian()

# Define Parameters
new_params = torch.view_as_complex(torch.randn(1,2))
new_params[0] = 1 + 0 * 1j
# new_params[1] = 0 + 0 * 1j

# Update model parameteres
model.update_params(new_params)

# Create a spacial grid object
grid = utils.PointGrid(100, start=-8, end=8)
                                    
# device tells where to store the tensor
mesh = grid.get_points().to(device)

# torch.unsqueeze(1) creates a [Nx, 1] tensor, ie, a batch of N items of size 1
# torch.requires_grad_() tells autograd to record operations on this tensor
x_grid = mesh.clone().unsqueeze(1).requires_grad_().type(torch.complex128)

#%% Stochastic Reconfiguration
# Initial conditions
pm.x0 = 1
pm.w = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 10

# Integrator parameters
pm.evolution = 'imag'

# Perform imag time evolution
imag_file_path = 'model_states_imag_evo.h5'
integrator(model, x_grid, file_path=imag_file_path)

# Get data and plot it
imag_evo = Dynamics(file_path=imag_file_path)
psi = imag_evo.compute_psi(x_grid)
plt.pcolor(imag_evo.t_grid, mesh, np.abs(psi.T)**2)

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

# Get data and plot it
real_evo = Dynamics(file_path=real_file_path)
psi = real_evo.compute_psi(x_grid)
plt.pcolor(real_evo.t_grid, mesh, np.abs(psi.T)**2)

#%%