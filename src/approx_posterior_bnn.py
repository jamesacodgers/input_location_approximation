import abc
import torch 

from src.layer_approx_posteriors import EnsembleLayer, MAPLayer, MFVILayer, SBVILayer


class BasePosterior(torch.nn.Module,abc.ABC):
    @abc.abstractmethod
    def __init__(self, likelihood: torch.distributions.Distribution, device):
        super().__init__()
        self.likelihood = likelihood
        self.device = device


    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def get_prior_contribution(self):
        pass

    @abc.abstractmethod
    def get_mean_log_likelihood(self, predictions, targets):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def sample_functions(self, x, n_samples):
        pass

    @abc.abstractmethod
    def get_CI(self, x, ci=0.95):
        pass

    @abc.abstractmethod
    def loss(self):
        pass

    def train_step(self, inputs, targets, optimizer):
        optimizer.zero_grad()
        loss = self.loss(inputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

class EnsemblePosterior(BasePosterior):
    def __init__(self, layer_priors, likelihood, device, num_samples):
        super(EnsemblePosterior,self).__init__(likelihood=likelihood, device=device)
        self.layers = torch.nn.ModuleList([EnsembleLayer(layer, n_samples=num_samples) for layer in layer_priors])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def get_CI(self, x, ci=0.95):
        with torch.no_grad():
            preds = self.forward(x)
            preds, idx = preds.sort(dim=0)
            lower_idx = int(((1-ci)/2)*preds.shape[0])
            upper_idx = int((1-(1-ci)/2)*preds.shape[0])
            lower_bound = preds[lower_idx]
            upper_bound = preds[upper_idx]
        return lower_bound, upper_bound

    def get_prior_contribution(self):
        """
        gets total prior contribution from all layers
        """
        total_prior = torch.zeros(1).to(self.device)
        for layer in self.layers:
            total_prior += layer.get_prior_contribution()
        return total_prior

    def get_mean_log_likelihood(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).mean(dim=0)
        
    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood(predictions, targets)
        return - (prior_contribution/targets.shape[0] + mean_log_likelihood_contribution)
    
    def sample_functions(self, x, n_samples):
        return self.forward(x)
        
class MAPPosterior(BasePosterior):
    def __init__(self, layer_priors, likelihood, device):
        super(MAPPosterior,self).__init__(likelihood=likelihood, device=device)
        self.layers = torch.nn.ModuleList([MAPLayer(layer) for layer in layer_priors])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def get_CI(self, x, ci=0.95):
        with torch.no_grad():
            preds = self.forward(x)
            return preds, preds  # MAP has no uncertainty

    def get_prior_contribution(self):
        """
        gets total prior contribution from all layers
        """
        total_prior = torch.zeros(1).to(self.device)
        for layer in self.layers:
            total_prior += layer.get_prior_contribution()
        return total_prior

    def get_mean_log_likelihood(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        
    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood(predictions, targets)
        return - (prior_contribution/targets.shape[0] + mean_log_likelihood_contribution)
    
    def sample_functions(self, x, n_samples):
        preds = self.forward(x)
        return preds.unsqueeze(0).expand((n_samples,preds.shape[0], preds.shape[1]))
        

class MFVIPosterior(BasePosterior):
    def __init__(self, layer_priors: list[MFVILayer], likelihood: torch.distributions.Distribution,  device: str, num_samples: int, temperature: float, posterior_exponentiation: str, n_data: int):
        super(MFVIPosterior,self).__init__(likelihood=likelihood, device=device)
        self.layers = torch.nn.ModuleList([MFVILayer(layer, num_samples=num_samples) for layer in layer_priors])
        
        self.posterior_exponentiation = posterior_exponentiation
        self.temperature = temperature
        self.activation = torch.nn.ReLU()
        self.n_data = n_data



    def forward(self, x, n_samples=None):
        x = x.unsqueeze(0)  # add sample dimension
        for layer in self.layers[:-1]:
            lin_update = layer(x, n_samples=n_samples)
            if layer.layer.in_features == layer.layer.out_features:
                x = self.activation(lin_update + x)
            else:
                x = self.activation(lin_update)
        x = self.layers[-1](x, n_samples=n_samples)
        return x

    def get_prior_contribution(self):
        """
        gets total prior contribution from all layers
        """
        total_prior = torch.zeros(1).to(self.device)
        for layer in self.layers:
            total_prior += layer.get_prior_contribution()
        return total_prior

    def get_mean_log_likelihood(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()
    
    def predict(self, x, n_samples=1024):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
        return preds.mean(dim=0)
    
    def sample_functions(self, x, n_samples=5):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
        return preds
    
    def get_CI(self, x, ci=0.95, n_samples=1024):

        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
            preds, idx = preds.sort(dim=0)
            lower_idx = int(((1-ci)/2)*preds.shape[0])
            upper_idx = int((1-(1-ci)/2)*preds.shape[0])
            lower_bound = preds[lower_idx]
            upper_bound = preds[upper_idx]
        return lower_bound, upper_bound
    
    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood(predictions, targets)
        # print(prior_contribution, mean_log_likelihood_contribution)
        if self.posterior_exponentiation == "tempered":
            return - (prior_contribution/self.n_data + 1/self.temperature*mean_log_likelihood_contribution)
        elif self.posterior_exponentiation == "cold":
            return - 1/self.temperature*(prior_contribution/self.n_data + mean_log_likelihood_contribution)
        
class WeightedMFVIPosterior(MFVIPosterior):
    def __init__(self, layer_priors: list[MFVILayer], likelihood: torch.distributions.Distribution,  device: str, num_samples: int, temperature: float, posterior_exponentiation: str, weighting_function: callable, n_data: int):
        super(WeightedMFVIPosterior,self).__init__( 
                                                   layer_priors=layer_priors, 
                                                   likelihood=likelihood,  
                                                   device=device, 
                                                   num_samples=num_samples, 
                                                   temperature=temperature, 
                                                   posterior_exponentiation=posterior_exponentiation
                                                   )
        self.weighting_function = weighting_function

    def get_likelihoods(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets)

    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        prior_contribution = self.get_prior_contribution()
        likelihoods = self.get_likelihoods(predictions, targets)
        weights = self.weighting_function(inputs)
        weighted_mean_likelihoods = (weights[None,...]*likelihoods).mean()
        if self.posterior_exponentiation == "tempered":
            return - (prior_contribution/targets.shape[0] + 1/self.temperature*weighted_mean_likelihoods)
        elif self.posterior_exponentiation == "cold":
            return - 1/self.temperature*(prior_contribution/targets.shape[0] + weighted_mean_likelihoods)
        

class SBVIPosterior(BasePosterior):
    def __init__(self, layer_priors: list[SBVILayer], likelihood: torch.distributions.Distribution,  device: str, num_samples: int, temperature: float, n_squash_vectors:int, posterior_exponentiation: str, n_data: int):
        super(SBVIPosterior,self).__init__(likelihood=likelihood, device=device)
        self.layers = torch.nn.ModuleList([SBVILayer(layer,n_squash_vectors=n_squash_vectors,  num_samples=num_samples) for layer in layer_priors])
        self.activation = torch.nn.ReLU()
        self._raw_std_scaling = torch.nn.Parameter(torch.ones(1)*1e-6)
        self.n_squash_vectors = n_squash_vectors
        n_params = torch.zeros(1)
        for layer in self.layers:
            n_params += torch.prod(torch.tensor(layer.mu_w.shape))
            n_params += torch.prod(torch.tensor(layer.mu_b.shape))
        self.n_params = torch.tensor([n_params]).to(device)
        self.posterior_exponentiation = posterior_exponentiation
        self.temperature = temperature
        self.n_data = n_data

    @property 
    def std_scaling(self): 
        return torch.abs(self._raw_std_scaling)
        # return torch.abs(self._raw_std_scaling)

    def forward(self, x, n_samples=None):
        x = x.unsqueeze(0)  # add sample dimension
        for layer in self.layers[:-1]:
            lin_update = layer(x, std_scaling= self.std_scaling, n_samples=n_samples)
            if layer.layer.in_features == layer.layer.out_features:
                x = self.activation(lin_update + x)
            else:
                x = self.activation(lin_update)
        x = self.layers[-1](x, std_scaling= self.std_scaling,n_samples=n_samples)
        return x

    def get_squash_matrix(self):
        squash_scaling = torch.zeros(self.n_squash_vectors, self.n_squash_vectors).to(self.device)
        for layer in self.layers:
            squash_scaling += layer.get_squashed_scale()

        return squash_scaling


    def get_prior_contribution(self):
        """
        gets total prior contribution from all layers
        """
        squash_scaling_matrix = self.get_squash_matrix()
        squash_eigvals = torch.linalg.eigvals(squash_scaling_matrix).real


        orthogonal_penalty = torch.sum(squash_scaling_matrix**2) - torch.sum(torch.diag(squash_scaling_matrix)**2)

        prior_var = self.layers[0].layer.weight_prior.variance[0,0].to(self.device)
        var_scaling = self.std_scaling**2
        squash_scaled_vars = (((1 - squash_eigvals)**2 ) * var_scaling)

        

        squared_mean = torch.zeros(1).to(self.device)
        for layer in self.layers:
            squared_mean += layer.get_mean_squared()

        det_ratio = self.n_params * torch.log(prior_var) - (self.n_params-self.n_squash_vectors)*torch.log(var_scaling) - torch.log( squash_scaled_vars).sum()

        trace_term = ((self.n_params-self.n_squash_vectors)*var_scaling + squash_scaled_vars.sum())/ prior_var

        squared_mean_term = squared_mean/prior_var

        kl = 0.5*(det_ratio - self.n_params + trace_term + squared_mean_term)
        
        return - kl  - orthogonal_penalty

    def get_mean_log_likelihood(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()
    
    def predict(self, x, n_samples=32):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
        return preds.mean(dim=0)
    
    def sample_functions(self, x, n_samples=16):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
        return preds
    
    def get_CI(self, x, ci=0.95, n_samples=32):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
            preds, idx = preds.sort(dim=0)
            lower_idx = int(((1-ci)/2)*preds.shape[0])
            upper_idx = int((1-(1-ci)/2)*preds.shape[0])
            lower_bound = preds[lower_idx]
            upper_bound = preds[upper_idx]
        return lower_bound, upper_bound
    
    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood(predictions, targets)
        # print(prior_contribution, mean_log_likelihood_contribution)
        if self.posterior_exponentiation == "tempered":
            return - (prior_contribution/self.n_data + 1/self.temperature*mean_log_likelihood_contribution)
        elif self.posterior_exponentiation == "cold":
            return - 1/self.temperature*(prior_contribution/self.n_data + mean_log_likelihood_contribution)
        
    def train_step(self, inputs, targets, optimizer):
        optimizer.zero_grad()
        loss = self.loss(inputs, targets)
        loss.backward()
        self._raw_std_scaling.grad.div_(self.n_params)
        # if self.n_squash_vectors > 0:
        #     for layer in self.layers:
        #         for param in layer.parameters():
        #             param.grad.div_(self.n_squash_vectors)
        optimizer.step()
        return loss