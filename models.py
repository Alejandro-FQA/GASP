import torch
import torch.nn as nn
import numpy as np

class Gaussian(nn.Module):
    '''
    Create a parametric Gaussian

    '''
    def __init__(self, num_params):
        super(Gaussian, self).__init__()
        # Initialize the parameters
        if num_params == 1 or 2:
            self.num_params = num_params
        else:
            self.num_params = 1
            print('num_params set to 1')
        self.params = nn.Parameter(torch.view_as_complex(torch.randn(self.num_params,2)))

    def update_params(self, new_params):
        # Optionally add a method to update the parameters manually
        with torch.no_grad():  # Make sure it doesn't interfere with autograd
            self.params.copy_(new_params)

    def forward(self, x):
        if self.num_params == 1:
            x0 = self.params[0].real
            p0 = self.params[0].imag
        else:
            x0 = self.params[0]
            p0 = self.params[1]
        
        return (1/torch.pi) ** 0.25 * torch.exp(-0.5 * (x - x0)**2) \
                              * torch.exp(1j * p0 * x) \
                              * torch.exp(-0.5j * x0 * p0)
        

