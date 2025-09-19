import abc
import torch 



class BaseLayer(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x, weights, bias):
        pass

    @abc.abstractmethod
    def get_prior_dist(self):
        ...

class LinearLayer(BaseLayer):
    def __init__(self, in_features: int, 
                 out_features: int, 
                 weight_prior: torch.distributions.Distribution,
                 bias_prior: torch.distributions.Distribution,
                 activation: torch.nn.Module,
                 ):
        """
        
        """
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior
        self.weight_shape = torch.Size([out_features, in_features])
        self.bias_shape = torch.Size([out_features])
        self.activation = activation


    def forward(self, x, weights, bias):
        assert weights.shape[0] == bias.shape[0]
        assert weights.shape[-2:] == self.weight_shape
        assert bias.shape[-1:] == self.bias_shape
        x = x@weights.transpose(-2,-1) + bias.unsqueeze(-2)
        return x

    def get_prior_dist(self):
        return self.weight_prior, self.bias_prior

    def activation(self, x):
        return self.activation(x)
    
    