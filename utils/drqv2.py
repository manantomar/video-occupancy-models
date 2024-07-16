from transformers import GPT2LMHeadModel, GPT2Config

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import utils
from pathlib import Path

from numpy_replay_buffer import EfficientReplayBuffer
from utils import load_offline_dataset_into_buffer

from dm_env import specs
import dmc


def k3s1p0(x):
    return x - 2


def k4s2p0(x):
    assert ((x % 2) == 0)
    return (x // 2) - 1


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        # x = T.Resize((x.shape[0], x.shape[1], 64, 64))
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 1

        action_dim = 6
        self.repr_dim = 75

        self.linear = nn.Sequential(nn.Linear(self.repr_dim, feature_dim), nn.BatchNorm1d(feature_dim),
                                   nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.linear(obs)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std=None):
        # h = self.trunk(obs)

        mu = self.policy(obs)
        mu = torch.tanh(mu)
        if std is None:
            return mu
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                  nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # h = self.trunk(obs)
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape=(75,), action_shape=(6,), device='cuda', lr=3e-4, feature_dim=64,
                 hidden_dim=256, critic_target_tau=0.005, num_expl_steps=2000,
                 update_every_steps=2, stddev_schedule='linear(1.0,0.1,100000)', 
                 stddev_clip=0.3, use_tb=False,
                 offline=True, bc_weight=2.5, augmentation=RandomShiftsAug(pad=4),
                 use_bc=True):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.offline = offline
        self.bc_weight = bc_weight
        self.use_bc = use_bc

        # replay buffer
        self.train_env = dmc.make("offline_cheetah_run_expert", 3, 2, 0, None)
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.discount = 0.99
        self.replay_buffer = EfficientReplayBuffer(25000, 32, 1, self.discount, 3, False, data_specs)

        offline_dir = "/home/manant/scratch/expert/vq_latents/"
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, None, 3, 25000)

        # gpt model
        configuration = GPT2Config(vocab_size=1024)
        self.gpt = GPT2LMHeadModel(configuration).to('cuda')
        self.gpt.load_state_dict(torch.load("/home/manant/scratch/pixel_gamma/checkpoints/vq_gpt_gamma_model_4000.pth"))

        # actor critic models
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(feature_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(feature_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(feature_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, latent, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0], None

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        if step % 5000 == 0:
            print("Critic Loss", critic_loss)

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = -Q.mean()

        actor_loss = actor_policy_improvement_loss

        # offline BC Loss
        if self.offline:
            actor_bc_loss = F.mse_loss(action, behavioural_action)
            # Eq. 5 of arXiv:2106.06860
            lam = self.bc_weight / Q.detach().abs().mean()
            if self.use_bc:
                actor_loss = actor_policy_improvement_loss * lam + actor_bc_loss
            else:
                actor_loss = actor_policy_improvement_loss #* lam

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.offline:
                metrics['actor_bc_loss'] = actor_bc_loss.item()

        if step % 5000 == 0:
            print("Actor Loss", actor_loss)

        return metrics

    def update(self, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch, indices = next(self.replay_buffer)
        obs, action, reward, discount, next_obs, _, _, _, _ = utils.to_torch(
            batch, self.device)

        # augment
        obs = obs.float() #self.aug(obs.float())
        next_obs = next_obs.float() #self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if self.offline:
            metrics.update(self.update_actor(obs.detach(), step, action.detach()))
        else:
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        if step % 20000 == 0:
            torch.save(self.actor.state_dict(), "/home/manant/scratch/pixel_gamma/checkpoints/drq_actor_{}.pth".format(step))
            torch.save(self.encoder.state_dict(), "/home/manant/scratch/pixel_gamma/checkpoints/drq_encoder_{}.pth".format(step))

        return metrics

if __name__ == "__main__":

    agent = DrQV2Agent()

    for step in range(2000000):
        agent.update(step)
