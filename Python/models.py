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
        

class NQS(nn.Module):
    '''
    Create a Neural Quantum State

    :input_size:        size of the input layer
    :output_size:       size of the output layer
    :hidden_layer:      list containing the size of each hidden layer
    :weights:           (optional) list of numpy arrays or tensors
    :biases:            (optional) list of numpy arrays or tensors
    :activavtion_fns:   (optional) list of activation functions - Sigmoid by default
    '''
    def __init__(self, input_size, output_size, hidden_layers, weights=None, biases=None, activation_fns=None):
        super(NQS, self).__init__()

        # Ensure activation_fns is provided and has the correct length
        if activation_fns is None:
            # activation_fns = [nn.Sigmoid] * len(hidden_layers)
            activation_fns = [StableSigmoid for _ in hidden_layers]
        elif len(activation_fns) != len(hidden_layers):
            raise ValueError("Length of activation_fns must match the length of hidden_layers")
        
        # List to hold all layers
        layers = []
        
        # Add the input layer and hidden layers
        prev_size = input_size
        for ii, (hidden_size, activation_fn) in enumerate(zip(hidden_layers, activation_fns)):
            # Create a linear layer from prev_size to hidden_size
            linear_layer = nn.Linear(prev_size, hidden_size, bias=(biases is None or biases[ii] is not None))
            layers.append(linear_layer)
            # Add activation function
            layers.append(activation_fn())
            # Update prev_size for the next layer
            prev_size = hidden_size
        # Add the output layer
        output_layer = nn.Linear(prev_size, output_size, bias=(biases is None or biases[len(hidden_layers)] is not None))
        layers.append(output_layer)
        # Combine layers into a sequential container
        self.network = nn.Sequential(*layers)
        
        # Optionally set custom weights and biases
        if weights and biases:
            self._initialize_weights_and_biases(weights, biases)
        else:
            weights, biases = self.initialize_weights_and_biases_xavier(input_size, hidden_layers, output_size)
            # Complex weights and biases
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                weights[i] = torch.view_as_complex(torch.stack((weight, weight * 0), -1))
                biases[i] = torch.view_as_complex(torch.stack((bias, bias * 0), -1))

            self._initialize_weights_and_biases(weights, biases)

    def initialize_weights_and_biases_xavier(self, input_size, hidden_layers, output_size):
        '''
        Initialize the weights and biases.

        The weights are initialized with the Xavier initialization, 
        which is is well-suited for networks with sigmoid activation functions,
        as it helps maintain stable gradients and activations throughout the layers. 
        '''      
        # Initialize weights
        weights = []
        biases = []

        # Input to first hidden layer
        w_input = torch.empty(hidden_layers[0], input_size, requires_grad=True)
        nn.init.xavier_uniform_(w_input)
        weights.append(w_input)
        
        b_input = torch.zeros(hidden_layers[0], requires_grad=True)
        biases.append(b_input)

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            w_hidden = torch.empty(hidden_layers[i+1], hidden_layers[i], requires_grad=True)
            nn.init.xavier_uniform_(w_hidden)
            weights.append(w_hidden)
            
            b_hidden = torch.zeros(hidden_layers[i+1], requires_grad=True)
            biases.append(b_hidden)
        
        # Last hidden layer to output
        w_output = torch.empty(output_size, hidden_layers[-1], requires_grad=True)
        nn.init.xavier_uniform_(w_output)
        weights.append(w_output)
        
        b_output = torch.zeros(output_size, requires_grad=True)
        biases.append(b_output)
        
        return weights, biases

    def _initialize_weights_and_biases(self, weights, biases):
        # Iterate through layers and set weights and biases
        all_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for ii, layer in enumerate(all_layers):
            layer.weight.data = weights[ii].clone().detach()
            if biases[ii] is not None:
                layer.bias.data = biases[ii].clone().detach()

    def forward(self, x):
        # Define the forward pass using the sequential container
        return self.network(x)

class StableSigmoid(nn.Module):
    def forward(self, input):  
        im = torch.remainder(input.imag, 2*torch.pi)
        re = input.real # torch.clamp(input.real, -88, 88)
        return torch.where(
                re >= 0,
                1 / (1 + torch.exp(-re) * torch.exp(-1j * im)),                                     # Standard sigmoid for positive x
                torch.exp(re) * torch.exp(1j * im) / (1 + torch.exp(re) * torch.exp(1j * im))       # Reformulated sigmoid for negative x
    )