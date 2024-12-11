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

# Choose model
pm.architecture = 'NQS'
match pm.architecture:
    case 'GASP':
        # Create the NN model
        num_params = 1 # only 1 or 2 parameters
        net_ark = f"{num_params}"
        model = Gaussian(num_params).to(device) 

        # Define Parameters
        new_params = torch.view_as_complex(torch.randn(num_params,2))
        # ground state values provided
        new_params[0] = 1 + 0 * 1j
        if num_params == 2:
            new_params[1] = 0 + 0 * 1j

        # Update model parameteres
        model.update_params(new_params)

    case 'NQS':
        # Network architecture
        input_size = 1
        output_size = 1
        hidden_layers = [16,16,16]
        net_ark = "-".join(map(str, [input_size, *hidden_layers, output_size]))        
        # Create Neural Quantum State
        model = NQS(input_size, output_size, hidden_layers).to(device)
        

# Model ID
file_name = lambda *args: "_".join(args)

# Create a spacial grid object
grid = utils.PointGrid(100, start=-8, end=8)
                                    
# device tells where to store the tensor
mesh = grid.get_points().to(device)

# torch.unsqueeze(1) creates a [Nx, 1] tensor
# torch.requires_grad_() tells autograd to record operations on this tensor
x_grid = mesh.clone().unsqueeze(1).requires_grad_().type(torch.complex128)

#%% Fitting initial condition
from tqdm import tqdm

learning_rate = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Create x_target as a leaf tensor
x_target = torch.tensor(mesh, dtype=torch.complex128, requires_grad=False).clone().unsqueeze(1)
# Target wavefunction
target = (1/np.pi)**(1/4) * np.exp(-0.5 * (x_target)**2)

def loss_fn(psi, target):
    # Complex Mean Squared Error (CMSE)
    real_loss = torch.mean((psi.real - target.real)**2)
    imag_loss = torch.mean((psi.imag - target.imag)**2)
    return real_loss + imag_loss

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e3, gamma=0.5)  # Reduce LR by 50% every 100 steps

# Training
loss_lst = []
epochs = 10000
with tqdm(total=epochs, disable=not pm.progress_bar) as pbar:
    for i in range(epochs):   
        # wavefunction
        psi = model(x_target)
        # Compute the loss
        loss = loss_fn(psi, target) 
        # Perform backpropagation and optimization
        optimizer.zero_grad()   # Initialize gradients to zero at each epoch
        loss.backward(retain_graph=False)         # Compute gradients
        optimizer.step()        # Update parameters
        
        # scheduler.step()  # Adjust learning rate

        # Optionally: print the loss for debugging
        pbar.set_postfix_str(f"Epoch {i}, Loss: {loss.item():.2e}")
        pbar.update(1)  

        loss_lst.append(loss.detach().numpy())

#%% Visualization
fig = plt.figure()  # Create the main figure

# Primary Y-Axis: Target and Model
ax = fig.add_subplot(111, facecolor="none")
ax.plot(mesh, np.abs(target.detach().numpy())**2, label='Target') 
ax.plot(mesh, np.abs(model(x_target).detach().numpy())**2, label='Model')
ax.set_xlabel('Mesh (Primary X-Axis)')  # Primary x-axis label
ax.set_ylabel('Amplitude Squared')  # Primary y-axis label
ax.legend(loc='upper right')

# Add Loss plot to the primary axis (optional)
ax2 = fig.add_subplot(111, facecolor="none")  # Secondary y-axis for loss
ax2.plot(range(epochs), loss_lst, label='Loss', color='red', linestyle='dashed')
ax2.set_yscale('log')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.set_xlabel('Epochs (Secondary X-Axis)')  # Label for the top x-axis
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Loss')  # Secondary y-axis label
ax2.legend(loc='right')

plt.show()

#%% Stochastic Reconfiguration
# Initial conditions
pm.x0 = 0
pm.w = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 50

# Integrator parameters
pm.evolution = 'imag'

# Perform imag time evolution
file_path = utils.file_ID(pm.data_dir,
                          file_name(pm.architecture, net_ark, pm.evolution),
                          ".h5")
integrator(model, x_grid, file_path=file_path)

# Get the dynamics
imag_evo = Dynamics(file_path=file_path, x_grid=x_grid)
# Compute density
den = np.abs(imag_evo.psi)**2
# Get parameters
params = imag_evo.get_params()

# Plot data
fig_path = utils.file_ID(pm.figs_dir,
                         file_name(pm.architecture, net_ark, pm.evolution),
                         ".png")
plots.evo_fig_params(imag_evo.t_grid, mesh, den.T, params, fig_path=fig_path)

#%% Stochastic Reconfiguration
# Initial conditions
pm.x0 = 0
pm.w = 1

# Time parameters
pm.dt = 0.1
pm.t_max = 50

# Integrator parameters
pm.evolution = 'real'

pm.lambda_reg = 1e-3 * (1 + 1j)

# Perform real time evolution
file_path = utils.file_ID(pm.data_dir,
                          file_name(pm.architecture, net_ark, pm.evolution),
                          ".h5")
integrator(model, x_grid, file_path=file_path)

# Get the dynamics
real_evo = Dynamics(file_path=file_path, x_grid=x_grid)
# Compute density
den = np.abs(real_evo.psi)**2
# Get parameters
params = real_evo.get_params()

# Plot data
fig_path = utils.file_ID(pm.figs_dir,
                         file_name(pm.architecture, net_ark, pm.evolution),
                         ".png")
plots.evo_fig_params(real_evo.t_grid, mesh, den.T, params, fig_path=fig_path)

# %% Compare with analytic results
# Analytic data
x0 = 1
p0 = 0

x = x0 * np.cos(real_evo.t_grid) + p0 * np.sin(real_evo.t_grid)
p = p0 * np.cos(real_evo.t_grid) - x0 * np.sin(real_evo.t_grid)

xx = mesh.numpy()[:, np.newaxis]

target = (1/np.pi)**(1/4) * np.exp(-0.5 * (xx - x)**2) \
                          * np.exp(1j * xx * p) \
                          * np.exp(-1j/2 * x * p) \
                          * np.exp(1j/2 * real_evo.t_grid*0) \
                          * np.sqrt(grid.get_spacing()) # match normalization
target_den = np.abs(target)**2 
target_energy = 0.5 * (1 + p**2 + x**2)

den_diff = den.T - target_den
energy_diff = real_evo.energy - target_energy

fig_path_0 = fig_path[:-4] + "_compare.png"
plots.evo_fig_compare(real_evo.t_grid, mesh, den_diff, energy_diff, fig_path=fig_path_0)

# %%
