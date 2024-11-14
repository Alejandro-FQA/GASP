''' [1] A. Sinibaldi et al., Quantum 7, 1131 (2023).'''

# PyTorch imports
import torch
from torch.autograd import grad

# Custom imports
import utils
import parameters as pm

# -----------------------------------------------------------------
# Function to compute energy expectation value
def energy(psi, grid, hamiltonian):

    # Split wavefunction
    u =       0.5 * (psi + psi.conj())
    v = -1j * 0.5 * (psi - psi.conj())  

    # Kinetic term
    (du_dx, _) = torch.view_as_real(utils.nth_derivative(u, grid, 1)).unbind(-1)
    (dv_dx, _) = torch.view_as_real(utils.nth_derivative(v, grid, 1)).unbind(-1)
    (d2u_d2x, _) = torch.view_as_real(utils.nth_derivative(du_dx, grid, 1)).unbind(-1)
    (d2v_d2x, _) = torch.view_as_real(utils.nth_derivative(dv_dx, grid, 1)).unbind(-1)

    d2psi_d2x = d2u_d2x + 1j * d2v_d2x

    kinetic = -(1/2) * d2psi_d2x    
    # Potential term
    potential = (1/2) * pm.w ** 2 * (grid - pm.x0).pow(2) * psi 
    # Applied Hamiltonian to psi    
    H_psi = kinetic + potential                                     
    energy = torch.einsum('ij,ij->j', psi, H_psi) / torch.einsum('ij,ij->j',psi, psi)
    return energy

# -----------------------------------------------------------------
# Function to compute variance of operator
def variance(psi, oper):
    "oper is a matrix already"
    oper2 = torch.matmul(oper, oper)
    x1 = torch.einsum('ik,ij,jk->k', psi.conj(), oper2, psi)
    x2 = torch.einsum('ik,ij,jk->k', psi.conj(), oper, psi) ** 2
    return (x1 - x2) / torch.einsum('ij,ij->j', psi.conj(), psi)

# -----------------------------------------------------------------
# Function to compute energy expectation value
def compute_energy(model, grid, hamiltonian):
    psi = model(grid)
    H_psi = hamiltonian(psi, grid)  # Apply Hamiltonian to psi
    energy = torch.vdot(psi[:,0], H_psi[:,0]) / torch.vdot(psi[:,0], psi[:,0])
    return energy

# -----------------------------------------------------------------
# Function to compute the Hamiltonian
def hamiltonian(psi, grid):    
    # Split wavefunction
    u =       0.5 * (psi + psi.conj())
    v = -1j * 0.5 * (psi - psi.conj())  

    # print("u has NaN:", torch.isnan(u).any())
    # print("u has inf or -inf:", torch.isinf(u).any())
    # print("v has NaN:", torch.isnan(v).any())
    # print("v has inf or -inf:", torch.isinf(v).any())

    # Kinetic term
    (du_dx, _) = torch.view_as_real(utils.nth_derivative(u, grid, 1)).type_as(u).unbind(-1)
    (dv_dx, _) = torch.view_as_real(utils.nth_derivative(v, grid, 1)).type_as(v).unbind(-1)
    (d2u_d2x, _) = torch.view_as_real(utils.nth_derivative(du_dx, grid, 1)).type_as(u).unbind(-1)
    (d2v_d2x, _) = torch.view_as_real(utils.nth_derivative(dv_dx, grid, 1)).type_as(v).unbind(-1)

    d2psi_d2x = d2u_d2x + 1j * d2v_d2x

    kinetic = -(1/2) * d2psi_d2x    
    # Potential term
    potential = (1/2) * pm.w ** 2 * (grid - pm.x0).pow(2) * psi  
    # Gaussian barrier
    gaussian = pm.gauss_amplitude * torch.exp(-(grid - pm.gauss_x0) ** 2 / 2 * pm.gauss_width ** 2) * psi

    return kinetic + potential + gaussian

# -----------------------------------------------------------------
# Function to compute the Jacobian of Ψ(x) wrt parameters
def compute_wirtinger_jacobian(model, outputs):
    '''
    Computes the Jacobian of the outputs of the model wrt the parameters of the model
    
    Arguments:
        model   (model): 
        inputs  (tensor): grid to feed the model

    Return:
        jacobian    (tensor): contains the derivatives of the outputs wrt the parameters of the model
    '''  

    # TRace anomalies
    # if pm.evo == 'real':
    #     torch.autograd.set_detect_anomaly(True) 

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
                                    grad_outputs=grad_outputs.unsqueeze(1).type(torch.complex128),
                                    # retain_graph=True,
                                    create_graph=True
                                )
                            ) for grad_outputs in torch.eye(u.shape[0]).unbind()]
                        )
                     ).unbind(-1)
    (dv_dx, dv_dy) = torch.view_as_real(
                        torch.stack(
                            [torch.nn.utils.parameters_to_vector(
                                grad(
                                    outputs=v,
                                    inputs=model.parameters(),
                                    grad_outputs=grad_outputs.unsqueeze(1).type(torch.complex128),
                                    # retain_graph=True,
                                    create_graph=True
                                )
                            ) for grad_outputs in torch.eye(v.shape[0]).unbind()]
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
