import abc
import torch 

from src.layer_approx_posteriors import MAPLayer


class BasePosterior(torch.nn.Module,abc.ABC):
    @abc.abstractmethod
    def __init__(self, total_data_points, batch_size):
        super().__init__()
        self.total_data_points = total_data_points
        self.batch_size = batch_size

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def get_prior_contribution(self):
        pass

    @abc.abstractmethod
    def get_mean_likelihood_contribution(self, predictions, targets):
        pass

    def loss(self, predictions, targets):
        prior_contribution = self.get_prior_contribution()
        mean_likelihood_contribution = self.get_mean_likelihood_contribution(predictions, targets)
        return - (prior_contribution + self.total_data_points*mean_likelihood_contribution.sum())

class MAPPosterior(BasePosterior):
    def __init__(self, layer_priors, log_likelihood, total_data_points, batch_size, device):
        super(MAPPosterior,self).__init__(total_data_points=total_data_points, batch_size=batch_size)
        self.layers = torch.nn.ModuleList([MAPLayer(layer) for layer in layer_priors])
        self.log_likelihood = log_likelihood
        self.device = device

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_prior_contribution(self):
        total_prior = torch.zeros(1).to(self.device)
        for layer in self.layers:
            total_prior += layer.get_prior_contribution()
        return total_prior

    def get_mean_likelihood_contribution(self, predictions, targets):
        return self.log_likelihood(predictions, targets).mean()
    
    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()