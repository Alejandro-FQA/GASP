# Architecture
architecture = 'NQS'

# Trapping potential
x0 = 0.
w = 1.

# Time evolution parameters
t_size = 1000           # time vector size without t = 0
dt = 0.1                # time discretization
t_max = 1               # last time instance
evolution = 'real'
integrator = 'RK4'
progress_bar = True

# Convergence parameters
stopper = True
e_error = 1e-4
steps = 10

# Parameteres to save data
data_dir = "./data/"
figs_dir = "./figs/"
fig_format = "pdf"
file_path = "model_states.h5"
overwrite = False
version = 1

# Regularization parameters
lambda_reg = 1e-3 * (1 + 1j*0)
reg = 'diagonal_shift'

# Gaussian barrier
gauss_width = 0
gauss_amplitude = 0
gauss_x0 = 0

# Mean field
g = -1
mu = 1