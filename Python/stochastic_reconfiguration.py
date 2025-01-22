''' [1] A. Sinibaldi et al., Quantum 7, 1131 (2023).'''

# PyTorch imports
import torch
from torch.autograd import grad
from torch.func import vmap

# Custom imports
import utilities as utils
import parameters as pm

def compute_energy(psi, x_grid):
    """
    Compute the energy of a state described by psi.
    psi is the output of self.model wiht input x_grid.

    Args:
        psi (torch.Tensor): Wavefunction.
        x_grid (torch.Tensor): Spatial grid points.

    Returns:
        energy (np.ndarray): Energy per particle.
    """
    H_psi = hamiltonian(psi, x_grid)  # Apply Hamiltonian to psi
    energy = torch.vdot(psi[:,0], H_psi[:,0]) / torch.vdot(psi[:,0], psi[:,0])
    return energy

# -----------------------------------------------------------------
# Function to compute the Hamiltonian
def hamiltonian(psi, grid): 

    # Kinetic term
    kinetic = -(1/2) * utils.second_derivative(psi, grid)  
    # kinetic = -(1/2) * utils.nth_derivative(psi, grid, 2)    
  
    # Potential term
    potential = (1/2) * pm.w ** 2 * (grid - pm.x0).pow(2) * psi  
    # Gaussian barrier
    gaussian = pm.gauss_amplitude * torch.exp(-(grid - pm.gauss_x0) ** 2 / 2 * pm.gauss_width ** 2) * psi

    # Mean-field
    mean_field = pm.g * torch.abs(psi).pow(2) * psi
    chem_pot = pm.mu * psi

    return kinetic + potential + gaussian + mean_field + chem_pot

# -----------------------------------------------------------------
# Function to compute the Jacobian of Ψ(x) wrt parameters
def old_compute_wirtinger_jacobian(model, outputs):
    '''
    Computes the Jacobian of the outputs of the model wrt the parameters of the model
    
    Arguments:
        model   (model): 
        outputs (tensor): grid to feed the model

    Return:
        jacobian    (tensor): contains the derivatives of the outputs wrt the parameters of the model
    '''  

    # Split wavefunction into real and imaginary
    u =       0.5 * (outputs + outputs.conj())
    v = -1j * 0.5 * (outputs - outputs.conj())

    # Compute partial derivatives - Jacobians
    (du_dx, du_dy) = torch.view_as_real(
                        torch.stack(
                            [torch.nn.utils.parameters_to_vector(
                                grad(
                                    outputs=u,
                                    inputs=model.parameters(),
                                    grad_outputs=grad_outputs.unsqueeze(1),
                                    # retain_graph=True,
                                    create_graph=True
                                )
                            ) for grad_outputs in torch.eye(u.shape[0]).type(torch.complex128).unbind()]
                        )
                     ).unbind(-1)
    (dv_dx, dv_dy) = torch.view_as_real(
                        torch.stack(
                            [torch.nn.utils.parameters_to_vector(
                                grad(
                                    outputs=v,
                                    inputs=model.parameters(),
                                    grad_outputs=grad_outputs.unsqueeze(1),
                                    # retain_graph=True,
                                    create_graph=True
                                )
                            ) for grad_outputs in torch.eye(v.shape[0]).type(torch.complex128).unbind()]
                        )
                     ).unbind(-1)
    
    # %% Wirtinger derivatives
    df_dx = du_dx + 1j * dv_dx
    df_dy = du_dy + 1j * dv_dy
    df_dz = 0.5 * (df_dx - 1j * df_dy)
    # df_dc = 0.5 * (df_dx + 1j * df_dy)

    # Return a complex Tensor of size [num_parameters, num_inputs]
    return df_dz.clone().detach()

# -----------------------------------------------------------------
# Function to compute the Jacobian of Ψ(x) wrt parameters
def compute_wirtinger_jacobian(model, outputs):
    """
    Computes the Jacobian of the outputs of the model wrt the parameters of the model.
    
    Arguments:
        model (nn.Module): PyTorch model containing parameters.
        outputs (Tensor): Complex-valued outputs from the model.

    Returns:
        Tensor: Wirtinger Jacobian (complex-valued).
    """
    # Split wavefunction into real and imaginary
    u =       0.5 * (outputs + outputs.conj())
    v = -1j * 0.5 * (outputs - outputs.conj())

    # grid length
    n = outputs.shape[0]

    # Compute partial derivatives - Jacobians
    def partial_derivatives(outputs, inputs):
        # Identity Tensor
        eye = torch.eye(n).type(torch.complex128).unsqueeze(-1)
        # Gradients
        gradients = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=eye,
            # retain_graph=True,
            create_graph=True,
            is_grads_batched=True
            )
        flattened_gradients = [gradient.reshape(n, -1) for gradient in gradients]
        return  torch.view_as_real(
                    torch.cat(flattened_gradients, dim=-1)
                    ).unbind(-1)

    # Partial derivatives
    (du_dx, du_dy) = partial_derivatives(u, model.parameters())
    (dv_dx, dv_dy) = partial_derivatives(v, model.parameters())
    
    # Wirtinger derivatives
    df_dx = du_dx + 1j * dv_dx
    df_dy = du_dy + 1j * dv_dy
    df_dz = 0.5 * (df_dx - 1j * df_dy)
    # df_dc = 0.5 * (df_dx + 1j * df_dy)

    # Return a complex Tensor of size [num_parameters, num_inputs]
    return df_dz.clone().detach()

# -----------------------------------------------------------------
# Function to compute the variational forces
def compute_variational_forces(psi, psi_grads, grid):
    '''
    Computes the variational forces

    Parameters:
        psi         (complex tensor): wavefunction
        psi_grads   (complex tensor): gradients of psi wrt parameters
        grid        (tensor): grid points
    
    Returns:
        variational_forces (complex tensor): variational forces
    '''
    # Evaluate the Hamiltonian
    H_psi = hamiltonian(psi, grid)    

    # Remove last dimension
    psi = psi.squeeze(1)
    H_psi = H_psi.squeeze(1)    

    # Conjugated quantities
    psi_conj = torch.conj(psi)
    psi_grads_conj = torch.conj(psi_grads)

    # Normalization
    N = torch.vdot(psi, psi)

    # Variational forces
    variational_forces =  torch.einsum('ji,j->i', psi_grads_conj, H_psi) / N - \
                          torch.einsum('ji,j,k,k->i', psi_grads_conj, psi, psi_conj, H_psi) / N ** 2
 
    return variational_forces.clone().detach()

# -----------------------------------------------------------------
# Function to compute the Quantum Geometric Tensor (QGT)
def compute_qgt(psi, psi_grads):
    '''
    Computes the Quantum Geometric Tensor

    Parameters:
    psi         (complex tensor): wavefunction
    psi_grads   (complex tensor): gradients of psi wrt parameters
    
    Returns:
    qgt         (complex tensor): quantum geometric tensor
    '''
    # Remove last dimension
    psi = psi.squeeze(1)

    # Conjugated quantities
    psi_conj = torch.conj(psi)
    psi_grads_conj = torch.conj(psi_grads)

    # Normalization
    N = torch.vdot(psi, psi)

    qgt = torch.einsum('ki,kj->ij', psi_grads_conj, psi_grads) / N - \
          torch.einsum('ji,j,k,kl->il', psi_grads_conj, psi, psi_conj, psi_grads) / N ** 2
   
    return qgt.clone().detach()

# -----------------------------------------------------------------
