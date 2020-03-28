import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    """(Variational-/Conditional-Variational-) AutoEncoder"""

    class Reshape(nn.Module):
        def __init__(self, dim:list):
            super().__init__()
            self.dim =  dim
            if len(dim) == 3:
                dim.insert(0, -1)
                self.dim = dim
        def forward(self, x):
            return x.view(self.dim)

    def __init__(self, in_shape, mlp_layer_sizes, latent_size=6, is_variational=False, conditional_size=0):
        super().__init__()
        assert type(mlp_layer_sizes) == list
        assert type(in_shape) == list

        k_size=[3,3]
        mlp_layer_sizes = [in_shape[0]] + mlp_layer_sizes
        e_modules , d_modules = [] , []

        x = torch.zeros([1] + list(in_shape))
        for i, (in_size, out_size) in enumerate(zip(mlp_layer_sizes[:-1], mlp_layer_sizes[1:])):
            s = 4 - i if i < 4 and i > 0 else 1
            p = [ k_size[0]//2 if x.shape[-2] < k_size[0] else 0,\
                  k_size[1]//2 if x.shape[-1] < k_size[1] else 0]
            e_modules.append( nn.Conv2d(in_size, out_size, k_size, padding=p, stride=s))
            x = e_modules[-1](x)
            e_modules.append(nn.LeakyReLU())
            d_modules.append(nn.ConvTranspose2d( out_size, in_size, k_size, padding=p, stride=s))
            d_modules.append(nn.LeakyReLU())

        mu_in_size = x.shape[-1] *  x.shape[-2] * x.shape[-3]

        e_modules.append(AutoEncoder.Reshape([-1, mu_in_size]))
        d_modules.append(AutoEncoder.Reshape([-1, x.shape[-3], x.shape[-2] , x.shape[-1]]))

        self.encoder = nn.Sequential(*e_modules)

        self.mu = nn.Sequential(nn.Linear(mu_in_size, latent_size))
        self.lvar = nn.Sequential(nn.Linear(mu_in_size, latent_size))

        latent_size = latent_size + conditional_size
        d_modules.append( nn.Linear(latent_size, mu_in_size) )

        d_modules= list(reversed(d_modules))
        d_modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential( *d_modules )

        self.is_variational = is_variational

    def forward(self, x, c=None):
        e = self.encoder(x)
        mu = self.mu(e)
        if not self.is_variational:
            return self.decoder(mu), mu, torch.zeros_like(mu)

        l_var = self.lvar(e)
        std = torch.exp(0.5*l_var)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, std
