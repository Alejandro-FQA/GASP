# Architecture
architecture = 'NQS'

# Trapping potential
x0 = 0.
w = 1.

# Time evolution parameters
t_size = 1000           # time vector size without t = 0
dt = 0.1                # time discretization
t_max = 50               # last time instance
evolution = 'imag'
integrator = 'RK4'
progress_bar = True

# Convergence parameters
stopper = True
e_error = 1e-5
steps = 10

# Parameteres to save data
data_dir = "./data/"
data_format = "h5"
figs_dir = "./figs/"
fig_format = "pdf"
file_path = "model_states.h5"
overwrite = True
version = 1

# Regularization parameters
lambda_reg = 1e-3 * (1 + 1j * 0)
reg = 'diagonal_shift'

# Gaussian barrier
gauss_width = 0
gauss_amplitude = 0
gauss_x0 = 0

# Mean field
g = 0
mu = 0