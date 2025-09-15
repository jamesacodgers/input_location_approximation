import abc
import torch 



class BaseLayer(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x, weights, bias):
        pass

class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, 
                 out_features: int, 
                 weight_prior: torch.distributions.Distribution,
                 bias_prior: torch.distributions.Distribution,
                 activation: torch.nn.Module):
        """
        
        """
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior
        self.weight_shape = torch.Size([out_features, in_features])
        self.bias_shape = torch.Size([out_features])
        self.activation = activation


    def forward(self, x, weights, bias):
        assert weights.shape == self.weight_shape
        assert bias.shape == self.bias_shape
        return torch.nn.functional.linear(x, weights, bias)
        
    def activation(self, x):
        return self.activation(x)
    
    