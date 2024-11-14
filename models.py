import torch
import torch.nn as nn
import numpy as np

class Gaussian(nn.Module):
    '''
    Create a parametric Gaussian

    '''
    def __init__(self):
        super(Gaussian, self).__init__()
        # Initialize the parameters
        self.params = nn.Parameter(torch.view_as_complex(torch.randn(1,2)))

    def update_params(self, new_params):
        # Optionally add a method to update the parameters manually
        with torch.no_grad():  # Make sure it doesn't interfere with autograd
            self.params.copy_(new_params)

    def forward(self, x):
        x0 = self.params[0].real
        p0 = self.params[0].imag

        # x0 = self.params[0]
        # p0 = self.params[1]
        
        return (1/torch.pi) ** 0.25 * torch.exp(-0.5 * (x - x0)**2) \
                              * torch.exp(1j * p0 * x) \
                              * torch.exp(-0.5j * x0 * p0)
        

