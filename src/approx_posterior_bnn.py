import abc
import torch 

from src.layer_approx_posteriors import EnsembleLayer, MAPLayer, MFVILayer


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
    def get_mean_log_likelihood_contribution(self, predictions, targets):
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

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output, y)
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

    def get_mean_log_likelihood_contribution(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()
    
    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).mean(dim=0)
        
    def loss(self, predictions, targets):
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood_contribution(predictions, targets)
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

    def get_mean_log_likelihood_contribution(self, predictions, targets):
        return self.likelihood.log_prob(predictions - targets).mean()
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        
    def loss(self, predictions, targets):
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood_contribution(predictions, targets)
        return - (prior_contribution/targets.shape[0] + mean_log_likelihood_contribution)
    
    def sample_functions(self, x, n_samples):
        preds = self.forward(x)
        return preds.unsqueeze(0).expand((n_samples,preds.shape[0], preds.shape[1]))
        

class MFVIPosterior(BasePosterior):
    def __init__(self, layer_priors: list[MFVILayer], likelihood: torch.distributions.Distribution,  device: str, num_samples: int, temperature: float, posterior_exponentiation: str):
        super(MFVIPosterior,self).__init__(likelihood=likelihood, device=device)
        self.layers = torch.nn.ModuleList([MFVILayer(layer, num_samples=num_samples) for layer in layer_priors])
        
        self.posterior_exponentiation = posterior_exponentiation
        self.temperature = temperature

    def forward(self, x, n_samples=None):
        x = x.unsqueeze(0)  # add sample dimension
        for layer in self.layers:
            x = layer(x, n_samples=n_samples)
        return x

    def get_prior_contribution(self):
        """
        gets total prior contribution from all layers
        """
        total_prior = torch.zeros(1).to(self.device)
        for layer in self.layers:
            total_prior += layer.get_prior_contribution()
        return total_prior

    def get_mean_log_likelihood_contribution(self, predictions, targets):
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
    
    def loss(self, predictions, targets):
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood_contribution(predictions, targets)
        if self.posterior_exponentiation == "tempered":
            return - (prior_contribution/targets.shape[0] + 1/self.temperature*mean_log_likelihood_contribution)
        elif self.posterior_exponentiation == "cold":
            return - 1/self.temperature*(prior_contribution/targets.shape[0] + mean_log_likelihood_contribution)