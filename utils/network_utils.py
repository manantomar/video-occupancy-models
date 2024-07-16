import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import utils

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

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


def get_parameter_names(model, forbidden_layer_types):
	"""
	Returns the names of the model parameters that are not inside a forbidden layer.
	"""
	result = []
	for name, child in model.named_children():
		result += [
			f"{name}.{n}"
			for n in get_parameter_names(child, forbidden_layer_types)
			if not isinstance(child, tuple(forbidden_layer_types))
		]
	# Add model specific parameters (defined with nn.Parameter) since they are not in any child.
	result += list(model._parameters.keys())
	return result

class Predictor(nn.Module):
	def __init__(self, feature_dim):
		super(Predictor, self).__init__()

		self.l1 = nn.Linear(feature_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.l3(a)


class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
	for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
		old_weight, up_weight = ma_params.data, current_params.data
		ma_params.data = ema_updater.update_average(old_weight, up_weight)


class ConvPredictor(nn.Module):
	def __init__(self, obs_shape):
		super().__init__()

		assert len(obs_shape) == 3
		self.repr_dim = 32 * 35 * 35
		feature_dim = 50
		hidden_dim = 1024

		self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
									 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
									 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
									 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
									 nn.ReLU())
		
		self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, 21))

		# self.linear = nn.Sequential(nn.Linear(self.repr_dim, 256), nn.ReLU(),
		# 							nn.Linear(256, 256), nn.ReLU(),
		# 							nn.Linear(256, 21), nn.Tanh())

		self.apply(utils.weight_init)

	def forward(self, obs, std=0.1, eval=False):
		obs = obs / 255.0 - 0.5
		h = self.convnet(obs)
		h = h.view(h.shape[0], -1)
		h = self.trunk(h)
		mu = self.policy(h)
		
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		if eval:
			action = dist.mean
		else:
			action = dist.sample(clip=0.3)
		return action

class projection_MLP(nn.Module):
	def __init__(self, in_dim, hidden_dim=256, out_dim=50): #256):
		super().__init__()
		# hidden_dim = in_dim
		self.layer1 = nn.Sequential(
			nn.Linear(in_dim, hidden_dim, bias=False),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Linear(hidden_dim, out_dim)
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		return x 

class InfoNCE(nn.Module):
	def __init__(self, feature_dim, action_dim, num_actions=1):
		super().__init__()
		
		self.train_samples = 256
		self.action_dim = action_dim

		self.projector = projection_MLP(feature_dim, 256, 1)

		# self.apply(weight_init)

	def forward(self, x1, x2, action, return_logits=False):
		self.device = x1.device
		# Generate N negatives, one for each element in the batch: (B, N, D).
		negatives = self.sample(x1.size(0), action.size(1))
		
		# Merge target and negatives: (B, N+1, D).
		targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)

		# Generate a random permutation of the positives and negatives.
		permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
		targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

		# Get the original index of the positive. This will serve as the class label
		# for the loss.
		ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

		# For every element in the mini-batch, there is 1 positive for which the EBM
		# should output a low energy value, and N negatives for which the EBM should
		# output high energy values.
		fused = torch.cat([x1.unsqueeze(1).expand(-1, targets.size(1), -1), x2.unsqueeze(1).expand(-1, targets.size(1), -1), targets], dim=-1)
		B, N, D = fused.size()
		fused = fused.reshape(B * N, D)
		out = self.projector(fused)
		energy = out.view(B, N)

		# Interpreting the energy as a negative logit, we can apply a cross entropy loss
		# to train the EBM.
		logits = -1.0 * energy
		loss = F.cross_entropy(logits, ground_truth.detach())

		if return_logits:
			return logits

		return loss

	def _sample(self, num_samples: int, action_size: int) -> torch.Tensor:
		"""Helper method for drawing samples from the uniform random distribution."""
		size = (num_samples, action_size)
		samples = np.random.uniform(-1, 1, size=size)
		return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

	def sample(self, batch_size: int, action_size: int) -> torch.Tensor:
		samples = self._sample(batch_size * self.train_samples, action_size)
		return samples.reshape(batch_size, self.train_samples, -1)

class MUSIKPredictor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(MUSIKPredictor, self).__init__()

		feature_dim = 50
		hidden_dim = 1024

		self.trunk = nn.Sequential(nn.Linear(state_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
							nn.ReLU(inplace=True),
							nn.Linear(hidden_dim, hidden_dim),
							nn.ReLU(inplace=True),
							nn.Linear(hidden_dim, action_dim))
		
		self.max_action = max_action
		self.apply(utils.weight_init)

	def forward(self, state, std=0.1, eval=False):
		h = self.trunk(state)
		mu = self.policy(h)
		mu = torch.tanh(mu)
		
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		if eval:
			action = dist.mean
		else:
			action = dist.sample(clip=0.3)
		return action


