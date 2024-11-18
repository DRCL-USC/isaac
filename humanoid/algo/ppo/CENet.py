import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

class CENet(nn.Module):
    def __init__(self, num_actor_obs):
        super(CENet, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.encoder = Encoder(num_actor_obs)
        self.decoder = Decoder(41)

    def forward(self,observations, real_v, real_p):
        temporal_partial_observations = observations[:, :self.num_actor_obs]
        estimation, latent_params = self.encoder(temporal_partial_observations)
        v, p,z = estimation
        next_partial_observations = self.decoder(real_v, real_p, z)
        return next_partial_observations

    def encode(self, observations):
        temporal_partial_observations = observations[:, :self.num_actor_obs]
        return self.encoder(temporal_partial_observations)

    def decode(self, vel, pos, z):
        return self.decoder(vel, pos, z)

    def loss_fn(self,obs_batch, next_obs, vel, pos, kl_weight=1.0):
        estimation, latent_params = self.encode(obs_batch)
        v, p, z = estimation
        vel_mu, vel_var, pos_mu, pos_var, latent_mu, latent_var = latent_params
        # Reconstruction loss
        recons = self.decode(vel, pos, z)
        recons_loss = F.mse_loss(recons, next_obs[:,self.num_actor_obs:], reduction='none').mean(-1)
        # Supervised loss
        vel_loss = F.mse_loss(v, vel, reduction='none').mean(-1)
        pos_loss = F.mse_loss(p, pos, reduction='none').mean(-1)

        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1)

        loss = recons_loss + vel_loss + kl_weight * kld_loss + pos_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'vel_loss': vel_loss,
            'kld_loss': kld_loss,
        }

class Encoder(nn.Module):
    def __init__(self, num_actor_obs):
        super(Encoder, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.elu = nn.ELU()
        self.l1 = nn.Linear(num_actor_obs,128)
        self.l2 = nn.Linear(128,64)
        # self.l3 = nn.Linear(64,22)
        self.v_mu = nn.Linear(64, 3)
        self.v_logvar = nn.Linear(64, 3)
        self.p_mu = nn.Linear(64, 3)
        self.p_logvar = nn.Linear(64, 3)
        self.z_mu = nn.Linear(64, 16)
        self.z_logvar = nn.Linear(64, 16)

    def forward(self, observations):
        x = self.elu(self.l1(observations))
        x = self.elu(self.l2(x))
        # x = self.l3(x)

        v_mu, v_logvar = self.v_mu(x), self.v_logvar(x)
        p_mu, p_logvar = self.p_mu(x), self.p_logvar(x)
        z_mu, z_logvar = self.z_mu(x), self.z_logvar(x)

        v = self.reparameterize(v_mu, v_logvar) #body velocity
        p = self.reparameterize(p_mu, p_logvar) #possition
        z = self.reparameterize(z_mu, z_logvar) #latent state

        return [v, p, z], [v_mu, v_logvar, p_mu, p_logvar, z_mu, z_logvar]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, num_actor_obs):
        super(Decoder, self).__init__()
        self.num_actor_obs = num_actor_obs
        self.elu = nn.ELU()
        self.l1 = nn.Linear(22,64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128,num_actor_obs)

    def forward(self, real_v, real_p, z):
        x = torch.cat((real_v, real_p, z), dim=1)
        x = self.elu(self.l1(x))
        x = self.elu(self.l2(x))
        next_partial_observations = self.l3(x)
        return next_partial_observations



