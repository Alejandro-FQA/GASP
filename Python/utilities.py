# PyTorch imports
import torch
from torch.autograd import grad

import numpy as np
import os

# Custom imports
import parameters as pm

def file_ID(directory, file_name, format):
    """
    Check if a file exists.
    If it exists and do not want to overwrite it, change name.

    Args:
        directory (str): path to directory.                    
        file_name (str): file name.
        format (str): file format.

    Returns:
        file_path (str): Path to file.
    """
    # TODO: fails when original file has been deleted
    file_path = directory + file_name + format
    v_id = pm.version
    msg = False
    # Check if original file exists
    if os.path.exists(file_path) and not pm.overwrite:
        file_path = directory + file_name + f"_v{v_id}" +  format
        msg = True
    # Check if version files exist
    while os.path.exists(file_path) and not pm.overwrite:
        v_id += 1
        file_path = directory + file_name + f"_v{v_id}" +  format
        msg = True

    pm.version = v_id
    if msg: print(f"This file version is v{v_id}")
    return file_path

def is_hermitian(A):
    return torch.allclose(A, A.conj().T)

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def derivative(f, x):
    """
    Compute the derivative of function f(x) at x.

    Args:
        f (torch.Tensor): Function that depends on x.                    
        x (torch.Tensor): Parameters of f.

    Returns:
        dfdx (torch.Tensor): Derivative of f at x.
    """
    try:
        dfdx, = grad(
                    outputs=f,
                    inputs=x,
                    grad_outputs=torch.ones_like(f).type_as(f),
                    create_graph=True
                    )
    except Exception as error:
        print(f"x_grid dtype: {x.dtype}, requires_grad: {x.requires_grad}")
        print(f"psi dtype: {f.dtype}, requires_grad: {f.requires_grad}")
        print('Error: ', error)

    return dfdx

def second_derivative(f, x):
    # Split function
    u =       0.5 * (f + f.conj())
    v = -1j * 0.5 * (f - f.conj())  

    (du_dx, _) = torch.view_as_real(derivative(u, x)).type_as(u).unbind(-1)
    (dv_dx, _) = torch.view_as_real(derivative(v, x)).type_as(v).unbind(-1)
    (d2u_d2x, _) = torch.view_as_real(derivative(du_dx, x)).type_as(u).unbind(-1)
    (d2v_d2x, _) = torch.view_as_real(derivative(dv_dx, x)).type_as(v).unbind(-1)

    return d2u_d2x + 1j * d2v_d2x

def nth_derivative(f, z, n):
    """
    Compute the n-th Wirtinger derivative of function f(z) wrt z.

    Args:
        f (torch.Tensor): Function that depends on z.                    
        z (torch.Tensor): Parameters of f.

    Returns:
        dnf_dnz (torch.Tensor): n-th derivative of f at z.
    """
    # Placeholder
    dnf_dnz = torch.empty_like(f)

    for _ in range(n):
        # Split function
        u =       0.5 * (dnf_dnz + dnf_dnz.conj())
        v = -1j * 0.5 * (dnf_dnz - dnf_dnz.conj())  

        # Partial derivatives
        (du_dx, du_dy) = grad(
                            outputs=u,
                            inputs=z,
                            grad_outputs=torch.ones_like(u).type_as(u),
                            create_graph=True
                            )
        (dv_dx, dv_dy) = grad(
                            outputs=v,
                            inputs=z,
                            grad_outputs=torch.ones_like(v).type_as(v),
                            create_graph=True
                            )
        # Wirtinger derivative
        dnf_dnz = 0.5 * (du_dx + 1j*dv_dx - 1j*du_dy + dv_dy)

    return dnf_dnz


def regularize_matrix(S, method="diagonal_shift", lambda_reg=0.01, clip_value=1.0):
    """
    Apply regularization to a matrix.
    
    Parameters:
    - S (torch.Tensor): Complex-valued matrix.
    - method (str): The regularization method to apply.
    - lambda_reg (float): Regularization strength.
    - clip_value (float): Value for clipping in weight clipping.
    
    Returns:
    - torch.Tensor: Regularized matrix.
    """
    # Ensure S is complex and square
    assert S.dtype == torch.cfloat or S.dtype == torch.cdouble, "S must be a complex tensor"
    assert S.size(0) == S.size(1), "S must be a square matrix"
    
    n = S.size(0)
    
    # Initialize regularized QGT as a copy of S
    S_reg = S.clone()

    # Using match-case to handle different regularization methods
    match method:
        case "diagonal_shift":
            # Diagonal Shift (Tikhonov Regularization)
            S_reg += lambda_reg * torch.eye(n, dtype=torch.cfloat)

        case "spectral":
            # Spectral Regularization
            U, singular_values, Vh = torch.svd(S)
            sigma_max = singular_values[0]  # Largest singular value
            spectral_loss = lambda_reg * sigma_max
            S_reg += spectral_loss * torch.eye(n, dtype=torch.cfloat)

        case "nuclear":
            # Nuclear Norm Regularization
            _, singular_values, _ = torch.svd(S)
            nuclear_loss = lambda_reg * singular_values.sum()
            S_reg += nuclear_loss * torch.eye(n, dtype=torch.cfloat)

        case "orthogonal":
            # Orthogonal Regularization
            S_hermitian = S.conj().T @ S
            identity = torch.eye(n, dtype=torch.cfloat)
            orthogonal_loss = lambda_reg * torch.norm(S_hermitian - identity, p='fro')**2
            S_reg += orthogonal_loss * torch.eye(n, dtype=torch.cfloat)

        case "frobenius":
            # Frobenius Norm Regularization
            frobenius_loss = lambda_reg * torch.norm(S, p='fro')**2
            S_reg += frobenius_loss * torch.eye(n, dtype=torch.cfloat)

        case "weight_clipping":
            # Weight Clipping
            S_reg = torch.clamp(S, min=-clip_value, max=clip_value)

        case "l1_norm":
            # L1 Norm Regularization (Sparse Regularization)
            l1_loss = lambda_reg * torch.sum(torch.abs(S))
            S_reg += l1_loss * torch.eye(n, dtype=torch.cfloat)

        case _:
            raise ValueError("Invalid regularization method specified.")

    return S_reg

def time_grid():
    # Time parameters
    Nt = pm.t_size          # time vector size without t = 0
    dt = pm.dt              # time discretization
    t_max = pm.t_max        # last time instance

    # check if we can provide enough time data points
    # otherwise create time grid accordingly
    if t_max / dt < Nt:
        return np.arange(0, t_max + dt, dt)   # time vector
    else:
        return np.linspace(0, t_max, Nt+1)    # time vector

class PointGrid:
    def __init__(self, N, spacing='linear', start=0, end=1, custom_points=None):
        """
        Initialize the PointGrid object.

        Args:
        - N (int): Number of points in the grid.
        - spacing (str): Type of spacing, either 'linear' or 'custom'. Default is 'linear'.
        - start (float): The starting value of the grid for linear spacing. Default is 0.
        - end (float): The ending value of the grid for linear spacing. Default is 1.
        - custom_points (list or None): List of custom points for non-linear spacing. Default is None.
        """
        self.N = N
        self.spacing = spacing
        self.start = start
        self.end = end
        self.custom_points = custom_points

        # Generate points based on the specified spacing type
        if self.spacing == 'linear':
            # Create N linear points between start and end
            self.points = torch.linspace(start, end, N)
        elif self.spacing == 'custom' and custom_points is not None:
            # Ensure custom_points length matches N and convert to tensor
            assert len(custom_points) == N, "Length of custom points must be equal to N"
            self.points = torch.tensor(custom_points)
        else:
            # Raise an error if spacing type is invalid or custom points are not provided
            raise ValueError("Invalid spacing type or custom points not provided")

    def get_points(self):
        """
        Get the grid points.

        Returns:
        - torch.Tensor: The grid points.
        """
        return self.points

    def get_limits(self):
        """
        Get the limits of the grid.

        Returns:
        - tuple: The start and end values of the grid.
        """
        return self.start, self.end

    def get_spacing(self):
        """
        Get the spacing between points.

        Returns:
        - float or None: The spacing value for linear spacing, or None for custom spacing.
        """
        if self.spacing == 'linear':
            return (self.end - self.start) / (self.N - 1)
        elif self.spacing == 'custom':
            return None  # Custom spacing doesn't have a uniform spacing value
        
    def get_weights(self):
        """
        Get the interation weights for each point.

        Returns:
        - torch.Tensor: The weights.
        """
        return torch.empty(self.N, 1).fill_(self.get_spacing())

    def get_properties(self):
        """
        Get all properties of the grid.

        Returns:
        - dict: A dictionary containing all grid properties.
        """
        return {
            'N': self.N,
            'spacing': self.spacing,
            'start': self.start,
            'end': self.end,
            'points': self.points
        }

