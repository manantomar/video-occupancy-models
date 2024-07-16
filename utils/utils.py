# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import h5py
from collections import deque
import dmc
from dm_env import StepType
from numpy_replay_buffer import EfficientReplayBuffer

import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import torchvision.transforms as T

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


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


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


step_type_lookup = {
    0: StepType.FIRST,
    1: StepType.MID,
    2: StepType.LAST
}


def load_offline_dataset_into_buffer(offline_dir, replay_buffer, agent, frame_stack, replay_buffer_size, future_sampling_steps=2, latent_style="vae"):
    filenames = sorted(offline_dir.glob('*.hdf5'))
    num_steps = 0
    print("filename is", filenames, offline_dir, offline_dir.glob('*.hdf5'))
    for filename in filenames:
        #try:
        episodes = h5py.File(filename, 'r')
        episodes = {k: episodes[k][:] for k in episodes.keys()}
        add_offline_data_to_buffer(episodes, replay_buffer, agent, framestack=frame_stack, future_sampling_steps=future_sampling_steps, latent_style=latent_style)
        length = episodes['reward'].shape[0]
        num_steps += length
        #except Exception as e:
        #    print(f'Could not load episode {str(filename)}: {e}')
        #    continue
        print("Loaded {} offline timesteps so far...".format(int(num_steps)))
        if num_steps >= replay_buffer_size:
            break
    print("Finished, loaded {} timesteps.".format(int(num_steps)))


def add_offline_data_to_buffer(offline_data: dict, replay_buffer: EfficientReplayBuffer, agent, framestack: int = 3, future_sampling_steps: int = 2, latent_style: str = "vae"):
    offline_data_length = offline_data['reward'].shape[0]
    for v in offline_data.values():
        assert v.shape[0] == offline_data_length
    done_list = np.argwhere(offline_data['step_type']==2)
    assert len(done_list) > 1 
    interval = done_list[1] - done_list[0]
    now = -1
    max_k = future_sampling_steps #15

    resize = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        # T.ToTensor()
    ])

    for idx in range(offline_data_length):
        time_step = get_timestep_from_idx(offline_data, idx, latent_style)
        if not time_step.first():
            now += 1
            # stacked_frames.append(np.asarray(resize(time_step.observation.reshape(84, 84, -1))).reshape(-1, 64, 64))
            stacked_frames.append(time_step.observation)
            stacked_pixel_frames.append(time_step.pixel_observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0), pixel_observation=np.concatenate(stacked_pixel_frames, axis=0))
            with torch.no_grad(): #, eval_mode(agent):
                ob = torch.as_tensor(np.concatenate(stacked_frames, axis=0), device='cuda')
                pixel_ob = torch.as_tensor(np.concatenate(stacked_pixel_frames, axis=0), device='cuda')
                # imp_action = torch.abs(agent.actor(agent.encoder(ob.unsqueeze(0)).squeeze(0)))
                #act = torch.as_tensor(imp_action.reshape(9, 84, 84), device=agent.device)
                #new_ob = torch.clamp(torch.sqrt(act) * ob + torch.sqrt(1 - act) * 255 * torch.randn(9, 84, 84, device=agent.device), min=0, max=255).type(torch.int64)
                # latent = agent.get_latent(ob, imp_action).squeeze(0) #agent.encoder(new_ob.unsqueeze(0)).squeeze(0) #agent.get_latent(state, action, latent=No)
            # time_step_stack = time_step_stack._replace(latent=latent.cpu().detach().numpy())
            # time_step_stack = time_step_stack._replace(imp_action=imp_action.cpu().numpy())
            rindex = min(interval-1, now+max_k)
            rindex = rindex - now
            time_step_stack = time_step_stack._replace(k_step=rindex)
            replay_buffer.add(time_step_stack)
        else:
            now = -1
            stacked_frames = deque(maxlen=framestack)
            stacked_pixel_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                # stacked_frames.append(np.asarray(resize(time_step.observation.reshape(84, 84, -1))).reshape(-1, 64, 64))
                stacked_frames.append(time_step.observation)
                stacked_pixel_frames.append(time_step.pixel_observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0), pixel_observation=np.concatenate(stacked_pixel_frames, axis=0))
            with torch.no_grad(): #, eval_mode(agent):
                ob = torch.as_tensor(np.concatenate(stacked_frames, axis=0), device='cuda')
                pixel_ob = torch.as_tensor(np.concatenate(stacked_pixel_frames, axis=0), device='cuda')
                # imp_action = torch.abs(agent.actor(agent.encoder(ob.unsqueeze(0)).squeeze(0)))
                #act = torch.as_tensor(imp_action.reshape(9, 84, 84), device=agent.device)
                #new_ob = torch.clamp(torch.sqrt(act) * ob + torch.sqrt(1 - act) * 255 * torch.randn(9, 84, 84, device=agent.device), min=0, max=255).type(torch.int64)
                # latent = agent.get_latent(ob, imp_action).squeeze(0) #agent.get_latent(state, action, latent=No)
                #imp_action = torch.abs(agent.actor(latent))
            # time_step_stack = time_step_stack._replace(latent=latent.cpu().detach().numpy())
            # time_step_stack = time_step_stack._replace(imp_action=imp_action.cpu().numpy())
            rindex = min(interval-1, now+max_k) #random.randint(now+1, min(interval-1, now+max_k))
            rindex = rindex - now
            time_step_stack = time_step_stack._replace(k_step=rindex)
            replay_buffer.add(time_step_stack)


## TODO: Add dummy values for storing 'mae_latents' and 'mae_ids_restore' so that a common replay buffer can be used for both, OR have two separate replay buffer codes
def get_timestep_from_idx(offline_data: dict, idx: int, latent_style: str):
    if latent_style == "vae":
        return dmc.ExtendedTimeStep(
            step_type=step_type_lookup[offline_data['step_type'][idx]],
            reward=offline_data['reward'][idx],
            pixel_observation=offline_data['observation'][idx],
            observation=offline_data['vae_latents'][idx],
            discount=offline_data['discount'][idx],
            action=offline_data['action'][idx],
            latent=np.zeros(256),
            imp_action=np.zeros(84*84*1),
            k_step = idx
        )
    elif latent_style == "dino":
        return dmc.ExtendedTimeStep(
            step_type=step_type_lookup[offline_data['step_type'][idx]],
            reward=offline_data['reward'][idx],
            pixel_observation=offline_data['observation'][idx],
            observation=offline_data['dino_latents'][idx],
            discount=offline_data['discount'][idx],
            action=offline_data['action'][idx],
            latent=np.zeros(256),
            imp_action=np.zeros(84*84*1),
            k_step = idx
        )