import abc
import torch

from src.layer_priors import BaseLayer


class BaseInferenceLayer(torch.nn.Module, abc.ABC):
    def __init__(self, layer: BaseLayer):
        super(BaseInferenceLayer,self).__init__()
        self.layer = layer


    @abc.abstractmethod
    def get_parameter_samples(self, model, x):
        pass

    def forward(self, x):
        weights, bias = self.get_parameter_samples(self.layer, x)
        out = self.layer.linear(x, weights, bias)
        out = self.layer.activation(out)
        return out
    
    @abc.abstractmethod
    def get_prior_contribution(self):
        pass

class MAPLayer(BaseInferenceLayer):
    def __init__(self, layer: BaseLayer):
        super(MAPLayer,self).__init__(layer)
        self.weight_map = torch.nn.Parameter(torch.randn(layer.weight_shape))
        self.bias_map = torch.nn.Parameter(torch.randn(layer.bias_shape))

    def get_parameter_samples(self):
        return self.weight_map, self.bias_map

    def get_prior_contribution(self):
        weight_log_prob = self.layer.weight_prior.log_prob(self.weight_map).sum()
        bias_log_prob = self.layer.bias_prior.log_prob(self.bias_map).sum()
        return weight_log_prob + bias_log_prob

    def forward(self, x):
        weights, bias = self.get_parameter_samples()
        out = self.layer.forward(x, weights, bias)
        out = self.layer.activation(out)
        return out
    
    
