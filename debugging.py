# %%
import torch

torch.set_default_dtype(torch.float64)

def get_prior_contribution(post_mean, squash_scaling_matrix, prior_var, std_scaling, n_params, n_squash_vectors):
        """
        gets total prior contribution from all layers
        """
        

        squash_eigvals = torch.linalg.eigvals(squash_scaling_matrix).real


        orthogonal_penalty = torch.sum(squash_scaling_matrix**2) - torch.sum(torch.diag(squash_scaling_matrix)**2)

        var_scaling = std_scaling**2
        squash_scaled_vars = (((1 - squash_eigvals)**2 ) * var_scaling)

        

        squared_mean = (post_mean**2).sum()


        det_ratio = n_params * torch.log(prior_var) - (n_params-n_squash_vectors)*torch.log(var_scaling) - torch.log( squash_scaled_vars).sum()

        trace_term = ((n_params-n_squash_vectors)*var_scaling + squash_scaled_vars.sum())/ prior_var

        squared_mean_term = squared_mean/prior_var

        kl = 0.5*(det_ratio - n_params + trace_term + squared_mean_term)

        return - kl 

p=2000
k=64
prior_var = torch.ones(1)*0.1

mu = torch.randn(p)
proj = torch.randn(k,p)

std_scaling = 0.1*torch.ones(1)


cov = (torch.eye(p) - proj.T @ proj).T @ (torch.eye(p) - proj.T @ proj) * (std_scaling**2) 


prior = torch.distributions.MultivariateNormal(torch.zeros(p), torch.eye(p)*prior_var)
post = torch.distributions.MultivariateNormal(mu, cov)


neg_kl = get_prior_contribution(mu, proj@proj.T, std_scaling=std_scaling, n_params=p, n_squash_vectors=k, prior_var=prior_var)

kl = torch.distributions.kl_divergence(post,prior)

print(neg_kl + kl)


# %% 
import matplotlib.pyplot as plt

n_samples = 500
correct_samples = post.sample((n_samples,))

def get_parameter_samples(mu_w, std_scaling, w_squash, n_params,  n_samples=n_samples):

        w_e = torch.randn([n_samples, n_params]) # [s, p, q]

        temp_w = torch.sum(w_e[:, None,:]*w_squash[None,:,:], dim=[-1]) # [s,r]
        temp_w = torch.sum(temp_w[:,:,None]*w_squash[None,:,:], dim=1) # [s, p] 

        w_e = w_e - temp_w
        weight_sample = mu_w[None,:] + std_scaling*w_e

        
        return weight_sample

gen_samples = get_parameter_samples(mu_w = mu, std_scaling=std_scaling, w_squash=proj,n_params=p)
# gen_samples_2 = mu + torch.randn(1000,p)@(torch.eye(p) - proj.T @ proj)

plt.scatter(correct_samples[:,0], correct_samples[:,1], label = "true", alpha=0.1)
plt.scatter(gen_samples[:,0], gen_samples[:,1], label = "gen", alpha=0.1)
# plt.scatter(gen_samples_2[:,0], gen_)
plt.legend()