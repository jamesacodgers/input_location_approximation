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
        return out
    
class EnsembleLayer(BaseInferenceLayer):
    def __init__(self, layer: BaseLayer, n_samples):
        super(EnsembleLayer,self).__init__(layer)
        self.weight_samples = torch.nn.Parameter(torch.randn(n_samples, layer.weight_shape))
        self.bias_samples = torch.nn.Parameter(torch.randn(n_samples, layer.bias_shape))

    def get_parameter_samples(self):
        return self.weight_samples, self.bias_samples

    def get_prior_contribution(self):
        weight_log_prob = self.layer.weight_prior.log_prob(self.weight_samples).mean()
        bias_log_prob = self.layer.bias_prior.log_prob(self.bias_samples).mean()
        return weight_log_prob + bias_log_prob

    def forward(self, x):
        weights, bias = self.get_parameter_samples()
        out = self.layer.forward(x, weights, bias)
        return out
    

class MFVILayer(BaseInferenceLayer):
    def __init__(self, layer: BaseLayer, num_samples=1):
        super(MFVILayer,self).__init__(layer)
        self.mu_w = torch.nn.Parameter(torch.randn(layer.weight_shape))
        self._raw_sigma_w = torch.nn.Parameter(-10*torch.ones(layer.weight_shape))
        self.mu_b = torch.nn.Parameter(torch.randn(layer.bias_shape))
        self._raw_sigma_b = torch.nn.Parameter(-10*torch.ones(layer.bias_shape))
        self.num_samples = num_samples

    def get_parameter_samples(self, n_samples=None):
        if n_samples is None:
            n_samples = self.num_samples
        weight_dist, bias_dist = self.get_approx_posteriors()
        weight_sample = weight_dist.rsample((n_samples,))
        bias_sample = bias_dist.rsample((n_samples,))
        return weight_sample, bias_sample

    def get_prior_contribution(self):
        """
        gets KL(q(w,b)||p(w,b)) where q is the variational distribution and p is the prior
        """
        weight_prior_dist, bias_prior_dist = self.layer.get_prior_dist()
        weight_approx_posterior, bias_approx_posterior = self.get_approx_posteriors()
        kl_weight = torch.distributions.kl_divergence(weight_approx_posterior, weight_prior_dist).sum()
        kl_bias = torch.distributions.kl_divergence(bias_approx_posterior, bias_prior_dist).sum()
        return - (kl_weight + kl_bias)

    def get_approx_posteriors(self):
        weight_approx_posterior = torch.distributions.Normal(self.mu_w, torch.nn.functional.softplus(self._raw_sigma_w))
        bias_approx_posterior = torch.distributions.Normal(self.mu_b, torch.nn.functional.softplus(self._raw_sigma_b))
        return weight_approx_posterior, bias_approx_posterior

    def forward(self, x, n_samples=None):
        if n_samples is not None:
            weights, bias = self.get_parameter_samples(n_samples=n_samples)
        else:
            weights, bias = self.get_parameter_samples()
        out = self.layer.forward(x, weights, bias)
        return out
    