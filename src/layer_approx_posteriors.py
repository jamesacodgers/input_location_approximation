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
        self.weight_samples = torch.nn.Parameter(torch.randn(n_samples, *layer.weight_shape))
        self.bias_samples = torch.nn.Parameter(torch.randn(n_samples, *layer.bias_shape))

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
        if layer.bias_prior is None:
            self.mu_b = None
            self._raw_sigma_b = None
        else: 
            self.mu_b = torch.nn.Parameter(torch.randn(layer.bias_shape))
            self._raw_sigma_b = torch.nn.Parameter(-10*torch.ones(layer.bias_shape))
        self.num_samples = num_samples

    def get_parameter_samples(self, n_samples=None):
        if n_samples is None:
            n_samples = self.num_samples
        weight_dist, bias_dist = self.get_approx_posteriors()
        weight_sample = weight_dist.rsample((n_samples,))
        if self.mu_b is None:
            bias_sample = (torch.ones(1)/(self.layer.bias_shape[0]/2)).to(weight_sample.device)  # dummy bias sample
        else:
            bias_sample = bias_dist.rsample((n_samples,))
        return weight_sample, bias_sample

    def get_prior_contribution(self):
        """
        gets KL(q(w,b)||p(w,b)) where q is the variational distribution and p is the prior
        """
        weight_prior_dist, bias_prior_dist = self.layer.get_prior_dist()
        weight_approx_posterior, bias_approx_posterior = self.get_approx_posteriors()
        kl_weight = torch.distributions.kl_divergence(weight_approx_posterior, weight_prior_dist).sum()
        if self.mu_b is None:
            kl_bias = torch.zeros(1).to(kl_weight.device)
        else:
            kl_bias = torch.distributions.kl_divergence(bias_approx_posterior, bias_prior_dist).sum()
        return - (kl_weight + kl_bias)

    def get_approx_posteriors(self):
        weight_approx_posterior = torch.distributions.Normal(self.mu_w, torch.nn.functional.softplus(self._raw_sigma_w))
        if self.mu_b is None:
            bias_approx_posterior = None
        else: 
            bias_approx_posterior = torch.distributions.Normal(self.mu_b, torch.nn.functional.softplus(self._raw_sigma_b))
        return weight_approx_posterior, bias_approx_posterior

    def forward(self, x, n_samples=None):
        if n_samples is not None:
            weights, bias = self.get_parameter_samples(n_samples=n_samples)
        else:
            weights, bias = self.get_parameter_samples()
        out = self.layer.forward(x, weights, bias)
        return out
    
class SBVILayer(BaseInferenceLayer):
    def __init__(self, layer: BaseLayer, num_samples=1):
        super(SBVILayer,self).__init__(layer)
        self.mu_w = torch.nn.Parameter(torch.randn(layer.weight_shape))
        self.w_squash = torch.nn.Parameter(1e-3*torch.ones(layer.weight_shape))

        self.mu_b = torch.nn.Parameter(torch.randn(layer.bias_shape))
        self.b_squash = torch.nn.Parameter(1e-3*torch.ones(layer.bias_shape))
        self.num_samples = num_samples

    def get_parameter_samples(self, std_scaling, squash_scaling, n_samples=None):
        if n_samples is None:
            n_samples = self.num_samples
        scaled_w_squash = self.w_squash/squash_scaling

        w_e = torch.randn([n_samples, *self.layer.weight_shape]).to(self.w_squash.device)
        w_e = w_e - torch.sum(w_e*scaled_w_squash[None,...], dim=[-1,-2], keepdim=True)*scaled_w_squash[None,...]
        weight_sample = self.mu_w + std_scaling*w_e

        scaled_b_squash = self.b_squash/squash_scaling
        b_e = torch.randn([n_samples, *self.layer.bias_shape]).to(self.b_squash.device)
        b_e = b_e - torch.sum(b_e*scaled_b_squash[None,...], dim=-1, keepdim=True)*scaled_b_squash[None,...]
        bias_sample = self.mu_b + std_scaling*b_e
        
        return weight_sample, bias_sample
    
    def get_squashed_scale(self):
        return torch.sum(self.b_squash**2) + torch.sum(self.w_squash**2)

    def get_prior_contribution(self):
        raise AssertionError("The prior contribution needs to be handled by the approx posterior parent class for SBVI")

    def forward(self, x, std_scaling, squash_scaling, n_samples=None):
        weights, bias = self.get_parameter_samples(std_scaling, squash_scaling, n_samples=n_samples)
        out = self.layer.forward(x, weights, bias)
        return out
    
    def get_mean_squared(self):
        return torch.sum(self.mu_b**2) + torch.sum(self.mu_w**2)
        