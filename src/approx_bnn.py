import abc
import torch 

from src.layer_approx_posteriors import MAPLayer, MFVILayer


class BasePosterior(torch.nn.Module,abc.ABC):
    @abc.abstractmethod
    def __init__(self, total_data_points: int, batch_size: int, likelihood: torch.distributions.Distribution, posterior_exponentiation: str, temperature: float):
        super().__init__()
        self.likelihood = likelihood
        self.total_data_points = total_data_points
        self.batch_size = batch_size
        self.posterior_exponentiation = posterior_exponentiation
        self.temperature = temperature

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
    def get_CI(self, x, ci=0.95):
        pass

    def loss(self, predictions, targets):
        prior_contribution = self.get_prior_contribution()
        mean_log_likelihood_contribution = self.get_mean_log_likelihood_contribution(predictions, targets)
        if self.posterior_exponentiation == "tempered":
            return - (prior_contribution + 1/self.temperature*self.total_data_points*mean_log_likelihood_contribution)
        elif self.posterior_exponentiation == "cold":
            return - 1/self.temperature*(prior_contribution + self.total_data_points*mean_log_likelihood_contribution)

class MAPPosterior(BasePosterior):
    def __init__(self, layer_priors, likelihood, total_data_points, batch_size, device):
        super(MAPPosterior,self).__init__(total_data_points=total_data_points, batch_size=batch_size, likelihood=likelihood)
        self.layers = torch.nn.ModuleList([MAPLayer(layer) for layer in layer_priors])
        self.device = device

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
    
    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        

class MFVIPosterior(BasePosterior):
    def __init__(self, layer_priors: list[MFVILayer], likelihood: torch.distributions.Distribution, total_data_points: int, batch_size: int, device: str, num_samples: int, temperature: float, posterior_exponentiation: str):
        super(MFVIPosterior,self).__init__(total_data_points=total_data_points, batch_size=batch_size, likelihood=likelihood, posterior_exponentiation=posterior_exponentiation, temperature=temperature)
        self.layers = torch.nn.ModuleList([MFVILayer(layer, num_samples=num_samples) for layer in layer_priors])
        self.device = device

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
    
    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def predict(self, x, n_samples=1024):
        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
        return preds.mean(dim=0)


    def get_CI(self, x, ci=0.95, n_samples=1024):

        with torch.no_grad():
            preds = self.forward(x, n_samples=n_samples)
            preds, idx = preds.sort(dim=0)
            lower_idx = int(((1-ci)/2)*preds.shape[0])
            upper_idx = int((1-(1-ci)/2)*preds.shape[0])
            lower_bound = preds[lower_idx]
            upper_bound = preds[upper_idx]
        return lower_bound, upper_bound