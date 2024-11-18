import torch
import h5py
import os
import io
import numpy as np

import stochastic_reconfiguration as SR

# -----------------------------------------------------------------
def save_model_architecture(model, file_path):
    """
    Saves the model architecture (scripted with torch.jit) in an HDF5 file.

    Args:
        model (torch.nn.Module): The model to save as a torch.jit.script file.
        file_path (str): Path to the HDF5 file to store the model.
    """
    # Script and save the model directly into the buffer
    temp_file = "temp_model.pt"
    scripted_model = torch.jit.script(model)
    scripted_model.save(temp_file)  # Save directly to the in-memory buffer
    
    # Read the saved file into memory and delete the temporary file
    with open(temp_file, "rb") as f:
        model_data = f.read()
    os.remove(temp_file)  # Ensure the temporary file is deleted

    # Save the binary data in the HDF5 file
    with h5py.File(file_path, 'a') as f:
        if "model_architecture" in f:
            del f["model_architecture"]  # Remove existing architecture if it exists
        # Save as a dataset with raw binary data
        f.create_dataset("model_architecture", data=np.void(model_data))
    # print(f"Model architecture saved in '{file_path}' under 'model_architecture'.")

# -----------------------------------------------------------------
def load_model_architecture(file_path):
    """
    Loads the model architecture from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file containing the model architecture.

    Returns:
        torch.jit.ScriptModule: The deserialized model architecture.

    Raises:
        KeyError: If the model architecture does not exist in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        if "model_architecture" not in f:
            raise KeyError(f"'model_architecture' not found in '{file_path}'")
        model_data = f["model_architecture"][()]

    # Use BytesIO to create an in-memory binary stream
    buffer = io.BytesIO(model_data.tobytes())
    
    # Load the model directly from the in-memory buffer using torch.jit
    loaded_model = torch.jit.load(buffer)
    # print(f"Model architecture loaded from '{file_path}'.")
    return loaded_model

# -----------------------------------------------------------------
def save_model_states(model, time_step, file_path):
    """
    Save model parameters in an HDF5 file.

    Args:
        model (torch.nn.Module): The model to save.
        time_step (int): The current time step identifier.
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'a') as f:
        group_name = f"time_{time_step}"        
        # Check if the group for this time step already exists
        if group_name in f:
            # If the group already exists, delete it to overwrite with new data
            del f[group_name]
        # Store the state dict at this time step
        group = f.create_group(f"time_{time_step}")
        for key, value in model.state_dict().items():
            group.create_dataset(key, data=value.cpu().numpy())
# -----------------------------------------------------------------
# def load_model_states(model, time_step, file_path):
#     """
#     Load model parameters, architecture, and additional parameters in an HDF5 file.

#     Args:
#         model (torch.nn.Module): The model to save.
#         time_step (int): The current time step identifier.
#         file_path (str): Path to the HDF5 file.
#     """
#     with h5py.File(file_path, 'r') as f:
#         group = f[f"time_{time_step}"]
#         state_dict = {key: torch.tensor(group[key]) for key in group.keys()}
#     model.load_state_dict(state_dict)

# -----------------------------------------------------------------
def save_variable(variable, name, file_path):
    """
    Save a variable as a dataset in an HDF5 file.

    Args:
        variable (np.ndarray or list): Array to save.
        name (str): Name of the variable.
        file_path (str): Path to the HDF5 file to store t_grid.
    """
    with h5py.File(file_path, 'a') as f:
        # Delete existing variable if it already exists to allow overwriting
        if name in f:
            del f[name]
        
        # Create a new dataset for variable
        f.create_dataset(name, data=variable)
    # print(f"'{name}' saved as a dataset in '{file_path}'.")

# -----------------------------------------------------------------
def load_variable(name, file_path):
    """
    Load the t_grid vector (time steps) from an HDF5 file.

    Args:
        name (str): Name of the variable.
        file_path (str): Path to the HDF5 file containing variable.

    Returns:
        np.ndarray: The loaded variable array.
        
    Raises:
        KeyError: If the specified variable name does not exist in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        if name not in f:
            raise KeyError(f"'{name}' not found in {file_path}")
        
        variable = f[name][:]
    # print(f"'{name}' loaded from {file_path}")
    return variable

# -----------------------------------------------------------------
# TODO
def save_variables(model, file_path):
    """
    Save the model architecture in an HDF5 file.

    Args:
        model (torch.nn.Module): The model to save.
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'a') as f:
        # Save additional parameters if they haven't been saved yet
        if "parameters" not in f:
            f.create_group("parameters")
            for attr in dir(pm):
                if not attr.startswith("__"):  # Skip special attributes
                    f["parameters"].attrs[attr] = getattr(pm, attr)

# -----------------------------------------------------------------
class Dynamics:
    def __init__(self, file_path, x_grid):
        """
        Initializes the Dynamics class from a saved model in an HDF5 file.
        
        Args:
            file_path (str): Path to the HDF5 file with the model.
            x_grid (torch.tensor): Spatial grid points.
        """
        # Load model
        self.file_path = file_path
        self.model = load_model_architecture(file_path)

        # Grids
        self.t_grid = load_variable("t_grid", file_path) 
        self.x_grid = x_grid

        # Main quantities
        self.psi, self.norm, self.energy = self.compute_psi()

    def load_model_state(self, time_step):
        """
        Load the model at a given time step.

        Args:
            time_step (int): The time step identifier to load.  
        """
        with h5py.File(self.file_path, 'r') as f:
            group = f[f"time_{time_step}"]
            state_dict = {key: torch.tensor(group[key]) for key in group.keys()}
        self.model.load_state_dict(state_dict)

    def compute_psi(self, x_grid=None, time_step=None):
        """
        Compute the model output (psi) for a given x_grid.
        If no time_step is given, psi is returned for all time steps.

        Args:
            x_grid (torch.Tensor): Spatial grid points.
            time_step (int): The time step identifier to load.  

        Returns:
            psi (np.ndarray): Wavefunction at each grid point normalized to 1.
            norm (np.array): Norm at each time step.
            enrgy (np.array): Energy per particle.
        """
        psi = []
        norm = []
        energy = []

        if not x_grid:
            x_grid = self.x_grid

        if not time_step:
            for it in range(len(self.t_grid)):
                self.load_model_state(it)
                output = self.model(x_grid)
                
                norm.append(torch.vdot(output[:,0], output[:,0]).detach().numpy())
                psi.append((output[:,0] / np.sqrt(norm[it])).detach().numpy())
                energy.append(self.compute_energy(output, x_grid).detach().numpy())
        else:
            self.load_model_state(time_step)
            output = self.model(x_grid)
            norm.append(torch.vdot(output[:,0], output[:,0]).detach().numpy())
            psi.append((output[:,0] / np.sqrt(norm[time_step])).detach().numpy())
            energy.append(self.compute_energy(output, x_grid).detach().numpy())

        return np.array(psi), np.array(norm), np.array(energy)
    
    def compute_energy(self, psi, x_grid):
        """
        Compute the energy of a state described by psi.
        psi is the output of the model wiht input x_grid.

        Args:
            psi (torch.Tensor): Wavefunction.
            x_grid (torch.Tensor): Spatial grid points.

        Returns:
            energy (np.ndarray): Energy per particle.
        """
        H_psi = SR.hamiltonian(psi[:,0].unsqueeze(-1), x_grid)  # Apply Hamiltonian to psi
        energy = torch.vdot(psi[:,0], H_psi[:,0]) / torch.vdot(psi[:,0], psi[:,0])
        return energy
    
    def compute_variance(psi, x_grid):
        """
        Compute the variance of a state described by psi.
        Variance defined as: <x²> - <x>²

        Args:
            psi (torch.Tensor): Wavefunction.
            x_grid (torch.Tensor): Spatial grid points.

        Returns:
            variance (np.ndarray): Varaiance.
        """
        # Spatial grid operator
        oper = torch.diag(x_grid.squeeze(-1))
        oper2 = torch.matmul(oper, oper)
        x1 = torch.einsum('ik,ij,jk->k', psi.conj(), oper2, psi)
        x2 = torch.einsum('ik,ij,jk->k', psi.conj(), oper, psi) ** 2
        variance = (x1 - x2) / torch.einsum('ij,ij->j', psi.conj(), psi)
        return variance.detach().numpy()

    def get_params(self, time_step=None):
        """
        Collect the model parameters.
        If no time_step is given, parameters are returned for all time steps.

        Args:
            time_step (int): The time step identifier to load.  

        Returns:
            params (np.ndarray): Parameters of the model
        """
        params = []
        if not time_step:
            for it in range(len(self.t_grid)):
                self.load_model_state(it)
                params.append(self.model.params.detach().numpy().copy())
        else:
            self.load_model_state(time_step)
            params.append(self.model.params.detach().numpy())

        return np.array(params)