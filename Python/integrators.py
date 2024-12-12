# PyTorch imports
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np
from tqdm import tqdm
import os

# Custom imports
import utilities as utils
import analysis
import parameters as pm
import stochastic_reconfiguration as SR

def reshape_parameters(model, parameters):
    '''
    Reshape the parameters to fit the NN arquitecture
    
    Parameters:
    model (model): NN containing the parameters
    parameters  (tensor): 1D tensor with the values of the parameters

    Return:
    list_parameters (list): returns a list of tensors with the model arquitecture
    '''
    # list of parameters with the arquiteture
    list_parameters = []
    start = 0
    for shape in [tuple(p.shape) for p in model.parameters()]:
        end = start + np.prod(shape)
        list_parameters.append(parameters[start:end].reshape(shape)) 
        start += np.prod(shape)  

    return list_parameters

# -----------------------------------------------------------------

def EOM(parameters, model, grid):
    '''
    Equations of motion
    '''        
    # Compute wave funtion
    psi = model(grid) 

    # TODO: check if normalization improves and is worth the time
    # norm = torch.vdot(psi[:,0], psi[:,0]).detach()
    # psi = psi / torch.sqrt(norm)

    # Compute the Jacobian
    jacobian = SR.compute_wirtinger_jacobian(model, psi)
    # Conpute the variational forces
    F = SR.compute_variational_forces(psi, jacobian, grid)
    # Compute the QGT
    S = SR.compute_qgt(psi, jacobian)

    # QGT regularization
    # S = utils.regularize_matrix(S, pm.reg, pm.lambda_reg)
    S +=  utils.eye_like(S) * pm.lambda_reg

    # Moore-Penrose psudo-inverse    
    # qgt_inv = torch.linalg.pinv(Sr)

    # Solve linear system:
    # TODO: choose linear solver
    match pm.evolution:
        case 'real':
            parameters = torch.linalg.solve(S, -1j * F)
            # parameters = torch.linalg.lstsq(S, -1j * F)[0]
        case 'imag':
            parameters = torch.linalg.solve(S, -F)
            # parameters = torch.linalg.lstsq(S, -F)[0]

    return parameters

# -----------------------------------------------------------------
def integrator(model, x_grid, t_grid=None, file_path=pm.file_path):
    """
    Performs numerical integration

    Args:
        model (torch.nn.Module): The model whose parameters evolve in time.
        x_grid (torch.Tensor): The spatial grid (inputs) for the model.
        t_grid (torch.Tensor): The time steps to save the model.
        If not given, default will be used.
        fiel_path (str): Path to the HDF5 file where the model is saved.
    """
    # Ensure data directory exists
    if not os.path.exists(pm.data_dir):
        os.makedirs(pm.data_dir)

    # Save model architecture and time steps
    analysis.save_model_architecture(model, file_path)
    if not t_grid:
        t_grid = utils.time_grid()

    # Flatten parameters
    # TODO: compare methods to export and import parameters
    # u = torch.cat([p.view(-1) for p in model.parameters()])
    u = parameters_to_vector(model.parameters())

    # Time steps to compute before saving the data
    t_step = round(pm.t_max / len(t_grid) / pm.dt)

    # Save the model state at initial time step in HDF5
    analysis.save_model_states(model, time_step=0, file_path=file_path)

    if pm.stopper and pm.evolution == 'imag':
        e0 = SR.compute_energy(model(x_grid), x_grid)
        check_point = 0

    # For each data time
    with tqdm(total=len(t_grid)-1, disable=not pm.progress_bar) as pbar:
        for it in range(1, len(t_grid)):
            # For each time step
            for _ in range(t_step):
                # Choose integration method
                match pm.integrator:
                    case 'RK4':    
                        u = RK4(u, model, x_grid)  
                    case 'Euler':
                        u = Euler(u, model, x_grid)
                # Update model
                vector_to_parameters(u, model.parameters())          

            try:
                # Check for NaN
                if torch.isnan(model(x_grid)).any():
                    raise ValueError(f"NaN encountered in wavefunction at time step {it}. " +\
                                      "Check parameters: dt, lambda_reg, reg.")   
                else:
                    # Save the model state at this time step in HDF5
                    analysis.save_model_states(model, time_step=it, file_path=file_path)
                    # Check for convergence
                    if pm.stopper and pm.evolution == 'imag':
                        energy = SR.compute_energy(model(x_grid), x_grid)
                        e_diff  = abs(energy - e0)
                        pbar.set_postfix_str(f"Energy error: {e_diff:.2e}")
                        if e_diff <= pm.e_error:
                            check_point += 1
                            if check_point == pm.steps // t_step + 1:
                                t_grid = t_grid[:it+1]
                                print('Convergence reached')                                
                                break
                        else:
                            e0 = energy
                            check_point = 0
                    # Update progress bar
                    pbar.update(1)  

            except ValueError as e:
                print(e)
                t_grid = t_grid[:it]
                break

        analysis.save_variable(t_grid, "t_grid", file_path)
        analysis.save_variable(pbar.format_dict['elapsed'], 'cmp_time', file_path)
# -----------------------------------------------------------------
def RK4(u, model, grid):

    ku1 = EOM(u, model, grid)
    vector_to_parameters(u + 0.5 * ku1 * pm.dt, model.parameters())
    ku2 = EOM(u + 0.5 * ku1 * pm.dt, model, grid)
    vector_to_parameters(u + 0.5 * ku2 * pm.dt, model.parameters())
    ku3 = EOM(u + 0.5 * ku2 * pm.dt, model, grid)
    vector_to_parameters(u + ku3 * pm.dt, model.parameters())
    ku4 = EOM(u + ku3 * pm.dt, model, grid)

    return u + pm.dt * ( ku1 + 2 * ku2 + 2 * ku3 + ku4 ) / 6

# -----------------------------------------------------------------
def Euler(u, model, grid):

    ku1 = EOM(u, model, grid)
    return u + pm.dt * ku1

# -----------------------------------------------------------------

