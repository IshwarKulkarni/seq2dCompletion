import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_encoder_decoder_2d(in_shape:list, conv_layer_sizes:list, k=3):
    conv_layer_sizes = [in_shape[0]] + conv_layer_sizes
    e_modules , d_modules = [] , []

    x = torch.zeros([1] + list(in_shape)) # NCHW, with N  =1
    for i, (in_size, out_size) in enumerate(zip(conv_layer_sizes[:-1], conv_layer_sizes[1:])):
        s = 2 if i < 4 and i > 0 else 1
        e = nn.Conv2d(in_size, out_size, k, stride=s)
        d = nn.ConvTranspose2d( out_size, in_size, k, stride=s)
        if d(e(x)).shape != x.shape:
            d = nn.ConvTranspose2d( out_size, in_size, k + k//2, stride=s)
        e_modules += [e, nn.LeakyReLU()] if i != len(conv_layer_sizes) - 2 else [e]
        d_modules += [d, nn.LeakyReLU()] if i != len(conv_layer_sizes) - 2 else [d]
        x = e(x)
    return e_modules, d_modules, list(x.shape)[1:] # encoder out shape in CHW (no N)

class Reshape(nn.Module):
        def __init__(self, dim:list):
            super().__init__()
            self.dim =  dim
        def forward(self, x):
            return x.view(self.dim)

class AutoEncoder(nn.Module):
    """(Variational-/Conditional-Variational-) AutoEncoder"""
    def __init__(self, in_shape, conv_layer_sizes, enc_out_size=6, decoder_in_size=6, output_nonlin=nn.Sigmoid):
        super().__init__()
        assert type(conv_layer_sizes) == list
        assert type(in_shape) == list
        assert len(in_shape) == 3

        e_modules , d_modules, e_out_sz = get_encoder_decoder_2d(in_shape, conv_layer_sizes)
        enc_lin_sz = e_out_sz[-3] * e_out_sz[-2] * e_out_sz[-1]  # linear size of encoder output

        self.len_to_2dmaps = len(e_modules) # after this, only linear layers are added.

        e_modules += [Reshape([-1, enc_lin_sz]), nn.Linear(enc_lin_sz, enc_out_size)]
        d_modules += [Reshape([-1] + e_out_sz), nn.Linear(decoder_in_size, enc_lin_sz)]
        
        self.encoder = nn.Sequential(*e_modules)
        d_modules= list(reversed(d_modules)) 
        if output_nonlin:
            d_modules.append(output_nonlin())
        self.decoder = nn.Sequential( *d_modules )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_2d_featurs(self, x):
        for i , e in enumerate(self.encoder):
            if i < self.len_to_2dmaps:
                x = e(x)
        return x

class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, in_shape, conv_layer_sizes, z_size=6, output_nonlin=nn.Sigmoid):
        super().__init__(in_shape, conv_layer_sizes, 2*z_size, z_size, output_nonlin=output_nonlin)
        self.mu_size = z_size

    def get_mu_lvar(self, x):
        e = self.encoder(x)
        mu = e[:, self.mu_size]
        lvar = e[:, self.mu_size:]
        return mu, lvar

    def forward(self, x):
        mu, lvar = self.get_mu_lvar(x)
        std = torch.exp(0.5*lvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z)

class ConditionalVariationalAutoEncoder(AutoEncoder):
    def __init__(self, in_shape, conv_layer_sizes, z_size=6, c_size=10, output_nonlin=nn.Sigmoid):
        super().__init__(in_shape, conv_layer_sizes, z_size, z_size + c_size, output_nonlin)
        
    def forward(self, x, c):
        mu, lvar = self.get_mu_lvar
        std = torch.exp(0.5*lvar)
        z = mu + torch.randn_like(std) * std
        z = torch.cat((z, c), 1)
        return self.decoder(z)

class Recurrent(AutoEncoder):
    def __init__(self, in_shape, conv_layer_sizes, batch_size, 
                 rnn_z_size=7, out_t=3, seq_len=6, output_nonlin=nn.Sigmoid):
        super().__init__(in_shape[1:], conv_layer_sizes, rnn_z_size, seq_len, 
                         output_nonlin)
        assert len(in_shape) == 4
        self.recurrent = nn.LSTM(input_size=rnn_z_size, hidden_size=in_shape[0],
                                 num_layers=seq_len, batch_first=True)

        self.out_t = out_t
        self.in_t = in_shape[0]
        self.rnn_hidden_size = seq_len
        self.hidden_cell = (torch.zeros(seq_len, batch_size, seq_len).to('cuda'),
                            torch.zeros(seq_len, batch_size, seq_len).to('cuda'))

    def combine_NT(self, x):
        s = list(x.shape) 
        return x.reshape([s[0]*s[1]] + s[2:])

    def split_NT(self, z, t_size):
        s = list(z.shape)
        return z.reshape([s[0]//t_size] + [t_size] + s[1:])

    def forward(self, x):
        
        z = self.encoder(self.combine_NT(x))
        z = self.split_NT(z, self.in_t)

        hidden_cell = self.hidden_cell if self.training else None
        out, hidden_cell = self.recurrent(z, hidden_cell)
        out  = out[:, self.out_t:, :]
        out = self.combine_NT(out)
        y = self.decoder(out)
        y = self.split_NT(y, self.out_t)
        return y