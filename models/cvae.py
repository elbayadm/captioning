import torch
import torch.nn as nn
from torch.autograd import Variable


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
    return mu + torch.exp(log_var / 2) * eps



class VAE(nn.Module):
    """
    Variatonal AE (encoder only)
    """
    def __init__(self, input_dim, latent_dim, h_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim  #X
        self.latent_dim = latent_dim #Z
        self.h_dim = h_dim #Reduction to h

        self.l_reduce = nn.Linear(self.input_dim, self.h_dim)
        self.l_reduce_nonlin = nn.ReLU()
        self.l_mu = nn.Linear(self.h_dim, self.latent_dim)
        self.l_var = nn.Linear(self.h_dim, self.latent_dim)

    def init_weights(self):
        nn.init.xavier_normal(self.l_reduce.weight)
        self.l_reduce.bias.data.fill_(0)

        nn.init.xavier_normal(self.l_mu.weight)
        self.l_mu.bias.data.fill_(0)
        nn.init.xavier_normal(self.l_var.weight)
        self.l_var.bias.data.fill_(0)

    def encode(self, x): #Q(X)
        h = self.l_reduce(x)
        h = self.l_reduce_nonlin(h)
        z_mu = self.l_mu(h)
        z_var = self.l_var(h)
        return z_mu, z_var

    def forward(self, x):
        z_mu, z_var = self.encode(x)
        z = sample_z(z_mu, z_var)
        return z_mu, z_var, z


class VAE_recon(nn.Module):
    """
    Variatonal AE (with MLP decoder)
    """
    def __init__(self, input_dim, latent_dim, h_dim):
        super(VAE_recon, self).__init__()
        self.input_dim = input_dim  #X
        self.latent_dim = latent_dim #Z
        self.h_dim = h_dim #Reduction to h

        self.l_reduce = nn.Linear(self.input_dim, self.h_dim)
        self.l_reduce_nonlin = nn.ReLU()
        self.l_mu = nn.Linear(self.h_dim, self.latent_dim)
        self.l_var = nn.Linear(self.h_dim, self.latent_dim)

        self.l_remap = nn.Linear(self.latent_dim, self.h_dim)
        self.l_remap_nonlin = nn.ReLU()

        self.recon = nn.Linear(self.h_dim, self.input_dim)
        self.recon_nonlin = nn.Sigmoid()

    def init_weights(self):
        nn.init.xavier_normal(self.l_reduce.weight)
        self.l_reduce.bias.data.fill_(0)

        nn.init.xavier_normal(self.l_mu.weight)
        self.l_mu.bias.data.fill_(0)
        nn.init.xavier_normal(self.l_var.weight)
        self.l_var.bias.data.fill_(0)

        nn.init.xavier_normal(self.l_remap.weight)
        self.l_remap.bias.data.fill_(0)

        nn.init.xavier_normal(self.recon.weight)
        self.recon.bias.data.fill_(0)

    def encode(self, x): #Q(X)
        h = self.l_reduce(x)
        h = self.l_reduce_nonlin(h)
        z_mu = self.l_mu(h)
        z_var = self.l_var(h)
        return z_mu, z_var


    def reconstruct(self, z):
        h = self.l_remap(z)
        h = self.l_remap_nonlin(h)
        X = self.recon(h)
        X = self.recon_nonlin(X)
        return X

    def forward(self, x):
        z_mu, z_var = self.encode(x)
        z = sample_z(z_mu, z_var)
        x_recon = self.reconstruct(z)
        return z_mu, z_var, x_recon


    # Loss
    #  recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    #  kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    #  loss = recon_loss + kl_loss



class CVAE(nn.Module):
    """
    Conditional variatonal AE
    """
    def __init__(self, input_dim, condition_dim, latent_dim, h_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim  #X
        self.output_dim = condition_dim #Y / Labels
        self.latent_dim = latent_dim #Z
        self.h_dim = h_dim #Reduction to h
        self.l_reduce = nn.Linear(self.input_dim + self.output_dim, self.h_dim)
        self.l_reduce_nonlin = nn.ReLU()
        self.l_mu = nn.Linear(self.h_dim, self.latent_dim)
        self.l_var = nn.Linear(self.h_dim, self.latent_dim)
        self.l_remap = nn.Linear(self.latent_dim + self.output_dim, self.h_dim)
        self.l_remap_nonlin = nn.ReLU()
        self.recon = nn.Linear(self.h_dim, self.input_dim)
        self.recon_nonlin = nn.Sigmoid()

    def init_weights(self):
        nn.init.xavier_normal(self.l_reduce.weight)
        self.l_reduce.bias.data.fill_(0)
        nn.init.xavier_normal(self.l_mu.weight)
        self.l_mu.bias.data.fill_(0)
        nn.init.xavier_normal(self.l_var.weight)
        self.l_var.bias.data.fill_(0)
        nn.init.xavier_normal(self.l_remap.weight)
        self.l_remap.bias.data.fill_(0)
        nn.init.xavier_normal(self.recon.weight)
        self.recon.bias.data.fill_(0)

    def encode(self, x, c): #Q(X,c)
        inputs = torch.cat([x, c], 1) #[X, Y]
        h = self.l_reduce(inputs)
        h = self.l_reduce_nonlin(h)
        z_mu = self.l_mu(h)
        z_var = self.l_var(h)
        return z_mu, z_var

    def reconstruct(self, z, c):
        inputs = torch.cat([z, c], 1)
        h = self.l_remap(inputs)
        h = self.l_remap_nonlin(h)
        X = self.recon(h)
        X = self.recon_nonlin(X)
        return X

    def forward(self, x, c):
        z_mu, z_var = self.encode(x, c)
        z = sample_z(z_mu, z_var)
        x_recon = self.reconstruct(z, c)
        return z_mu, z_var, x_recon


    #  Loss
    #  recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    #  kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    #  loss = recon_loss + kl_loss

