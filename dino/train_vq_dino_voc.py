# from torchvision import transforms as pth_transforms
from timm.models import create_model
import modeling_vqkd 

import sys
sys.path.insert(0, '../utils/')

import os
import math
import copy
import wandb
import random
import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from numpy_replay_buffer import EfficientReplayBuffer
from utils import load_offline_dataset_into_buffer

# import dmc
from dm_env import specs

from transformers.optimization import get_scheduler
from transformers import GPT2LMHeadModel, GPT2Config

from network_utils import get_parameter_names, update_moving_average, Predictor, EMA, RandomShiftsAug

##################
obs_resize = 224

offline_dir = "../../cheetah_train/dino_latents/"
dino_path = '../../vqkd_encoder_base_decoder_1x768x12_dino-663c55d7.pth'
save_dir_path = "./"

batch_size = 32
frame_stack = 2
device = 'cuda'
##################
# - Pretrained DINO + Quantization 
##################

class TFAgent:
    def __init__(self, discount=0.8, augmentation=RandomShiftsAug(pad=4)):

        wandb.init(project="Video Occupancy Models",
               id="voc_dino_gamma_{}".format(discount),
               entity=None, dir=os.getcwd())

        data_specs = (specs.BoundedArray(shape=(9, 84, 84), dtype=np.uint8, name='observation', minimum=0, maximum=255),
                      specs.BoundedArray(shape=(6,), dtype=np.float32, name='action', minimum=-1.0, maximum=1.0),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.discount = discount
        self.batch_size = batch_size
        self.nsteps = min(int(1 / (1 - self.discount)), 10) # only sample max 10 steps
        self.codebook_size = 8192

        ######### train on VQ latents #########
        self.replay_buffer = EfficientReplayBuffer(1000000, self.batch_size, 1, self.discount, frame_stack, False, data_specs)
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, None, frame_stack, 1000000, latent_style="dino")
        ##########################

        self.model = create_model(
            'vqkd_encoder_base_decoder_1x768x12_dino',
            pretrained=True,
            pretrained_weight=dino_path,
            as_tokenzer=True,
        ) #.to('cuda').eval()
        self.model.to(device).eval()

        ######### gpt setup #########
        configuration = GPT2Config(vocab_size=8192, n_positions=196 * frame_stack * 2, n_layer=4, n_head=4, n_embed=128) # nano-gpt
        self.gpt = GPT2LMHeadModel(configuration).to(device)
        self.gpt_target = copy.deepcopy(self.gpt)
        self.gpt_target.generation_config.output_scores = True
        self.target_ema_updater = EMA(0.9)
        ##########################

        ######### optimizer setup #########
        self.reward_predictor = Predictor(32 * 14 * 14 * frame_stack).to(device)
        self.gpt_optimizer = torch.optim.AdamW(list(self.get_grouped_params(self.gpt)), lr=3e-4)
        self.optimizer = torch.optim.AdamW(list(self.reward_predictor.parameters()), lr=3e-4)

        num_training_steps = 100000
        self.warmup_ratio = 0.05
        warmup_steps = math.ceil(num_training_steps * self.warmup_ratio)
        self.lr_scheduler = get_scheduler(
                "cosine",
                optimizer=self.gpt_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        ##########################

        self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

        self.device = device
        self.aug = augmentation

        self.saving_iter = [50, 100, 500, 1000, 2000, 5000, 10000, 50000, 75000, 100000]
        self.train()

    def get_grouped_params(self, model):
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def train(self, training=True):
        self.training = training

    def preprocess_obs(self, obs):
        obs = F.interpolate(obs, size=obs_resize)    
        
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        try:
            assert len(obs.shape) == 4 # B x C x H x W
            org_obs_shape = obs.shape
            # normalize and preprocess
            obs = torch.stack([torch.einsum('nchw->nhwc', obs[:, i*3:3+i*3] / 255.) - self.imagenet_mean / self.imagenet_std for i in range(frame_stack)])
            obs = torch.einsum('snhwc->nschw', obs).reshape((org_obs_shape[0] * frame_stack, 3, *org_obs_shape[2:]))
        except:
            assert len(obs.shape) == 5 # T x B x C x H x W
            org_obs_shape = t, b, c, h, w = obs.shape
            obs = torch.stack([torch.einsum('tnchw->tnhwc', obs[:, :, i*3:3+i*3] / 255.) - self.imagenet_mean / self.imagenet_std for i in range(frame_stack)])
            obs = torch.einsum('stnhwc->tnschw', obs).reshape((t, b * frame_stack, 3, h, w))
        return obs, org_obs_shape

    def update(self, step=0):
        metrics = dict()

        batch, indices = next(self.replay_buffer)
        obs, action, reward, discount, next_obs, _, _, _, obs_k = utils.to_torch(batch, self.device)

        y_context = obs.long()
        y_target = obs_k.long()
        obs_shape = obs.shape

        quant_target = self.model.get_codebook_entry(y_target.reshape(obs_shape[0]*frame_stack, -1))

        quant_context = self.model.get_codebook_entry(y_context.reshape(obs_shape[0]*frame_stack, -1))

        pred_reward = self.reward_predictor(quant_target.detach().float().reshape(obs_shape[0], -1))
        reward_loss = F.mse_loss(pred_reward, reward.float()).mean()

        # generate target
        with torch.no_grad():
            p_t = self.gpt_target.generate(y_target, max_new_tokens=y_target.shape[-1], do_sample=True, pad_token_id=-100)
            p_t = p_t[:, -y_target.shape[-1]:]

        # gamma sampling
        gamma = self.discount * torch.ones((y_context.shape[0], ), device=y_context.device)
        prob = torch.bernoulli(gamma)
        p_target = torch.zeros_like(y_target)

        # with prob 1-gamma, sample from next state
        p_c_idx = torch.nonzero(1 - prob)
        p_target[p_c_idx] = y_target[p_c_idx]

        # with prob gamma, sample from bootstrapped model
        p_t_idx = torch.nonzero(prob)
        p_target[p_t_idx] = p_t[p_t_idx]

        # gpt predictions
        inp = torch.cat([y_context, p_target], dim=1)
        # mask_ids = torch.cat([context_mask_ids, target_mask_ids], dim=1)
        outputs = self.gpt(inp, labels=inp)
        gpt_loss = outputs.loss

        loss = gpt_loss + reward_loss
        loss.backward()
        
        # grad accumulate
        if step % 1 == 0:
            self.optimizer.step()
            self.gpt_optimizer.step()

            self.optimizer.zero_grad()
            self.gpt_optimizer.zero_grad()

            self.lr_scheduler.step()
            update_moving_average(self.target_ema_updater, self.gpt_target, self.gpt)

        # visualize predictions 
        if step % 200 == 0:
            with torch.no_grad():
                # sample a batch of traj and corresponding values
                batch, indices = self.replay_buffer.sample_spr(jumps=self.nsteps)
                _, _, _, _, _, all_obs, all_pixel_obs, _, values = utils.to_torch(batch, self.device)
                
                # preprocess first obs from traj
                obs = all_obs[0]
                obs_shape = obs.shape
                
                # embed first obs
                y_context = obs.long()

                value_loss = self.get_value_estimates(y_context, values, obs_shape)
                wandb.log({"value loss": value_loss}, step=step)

        wandb.log({"gpt loss": gpt_loss}, step=step)
        wandb.log({"reward loss": reward_loss}, step=step)

        # save gpt model
        if step in self.saving_iter:
            print("saving gpt weights...")
            self.save_gpt_weights(step)

        return metrics

    def save_gpt_weights(self, step):
        torch.save(self.gpt.state_dict(), os.path.join(save_dir_path, "dino_nanogpt_gamma_{}_{}_model_step_{}.pth".format(self.discount, self.codebook_size, step)))

    def get_value_estimates(self, y_context, values, obs_shape):
        # Take a state, get samples from the gamma distribution, 
        # Run the reward predictor through these to get value estimates
        # Get ground truth value estimates by simply taking discounted sum of rewards
        # Compare these for different states
        
        num_gamma_samples = 100
        values_pred = []
        
        for i in range(num_gamma_samples):
            outputs = self.gpt_target.generate(y_context, max_new_tokens=y_context.shape[-1], do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=-100) #, kwargs={'token_type_ids': context_mask_ids})
            p_t = outputs.sequences[:, -y_context.shape[-1]:]
            
            # quant = self.model.quantize.get_codebook_entry(p_t, None)
            # quant = quant.view(-1, 5, 5, 256).permute(0, 3, 1, 2)
            quant = self.model.get_codebook_entry(p_t.reshape(obs_shape[0]*frame_stack, -1))
            
            values_pred.append(self.reward_predictor(quant.float().reshape(obs_shape[0], -1)).squeeze(1))

        values_pred = torch.stack(values_pred).sum(0) / (100 * (1 - self.discount))

        value_estimation_loss = F.mse_loss(values_pred, values.squeeze(1).float()).mean()
        print("val estimation", value_estimation_loss, values_pred[:5], values[:5])
        
        return value_estimation_loss


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a dictionary with command-line arguments.')
    
    parser.add_argument('--discount', type=float, default=0.8, help='discount')
    args = parser.parse_args()

    agent = TFAgent(discount=args.discount)

    agent.optimizer.zero_grad()
    agent.gpt_optimizer.zero_grad()

    for step in range(100000):
        agent.update(step)