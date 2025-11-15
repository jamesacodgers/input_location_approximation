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
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation


    def forward(self, x, weights, bias):
        assert weights.shape[0] == bias.shape[0]
        assert weights.shape[-2:] == self.weight_shape
        assert bias.shape[-1:] == self.bias_shape
        lin_out = x@weights.transpose(-2,-1) + bias.unsqueeze(-2)
        if self.in_features == self.out_features:
            return self.activation(lin_out + x )
        return self.activation(lin_out)

    def get_prior_dist(self):
        return self.weight_prior, self.bias_prior


    
class FourierLayer(BaseLayer):
    def __init__(self, 
                 in_features: int,
                 out_features: int, 
                 weight_prior: torch.distributions.Distribution,
                 bias_prior: torch.distributions.Distribution,
                 ):
        """
        """
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior
        self.weight_shape = torch.Size([out_features, in_features])
        self.bias_shape = torch.Size([out_features])

    def get_prior_dist(self):
        return self.weight_prior, self.bias_prior

    def forward(self, x, weights, bias):
        cos_x = bias.unsqueeze(-2)*torch.cos(2*torch.pi*x@weights.transpose(-2,-1))
        sin_x = bias.unsqueeze(-2)*torch.sin(2*torch.pi*x@weights.transpose(-2,-1))
        x = torch.cat([cos_x, sin_x], dim=-1)
        return x    
   