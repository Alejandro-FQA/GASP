
# Trapping potential
x0 = 0.
w = 1.

# Time evolution parameters
t_size = 1000           # time vector size without t = 0
dt = 0.1                # time discretization
t_max = 1               # last time instance
evolution = 'real'
integrator = 'RK4'

# Path to save data
file_path = "model_states.h5"

# Regularization parameters
lambda_reg = 1e-8 * (1 + 1j)
reg = 'diagonal_shift'

# Gaussian barrier
gauss_width = 0
gauss_amplitude = 20
gauss_x0 = 0
