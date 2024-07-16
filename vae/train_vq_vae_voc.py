import sys
sys.path.insert(0, '../utils/')

from transformers.optimization import get_scheduler
from transformers import GPT2LMHeadModel, GPT2Config

from taming.models.vqgan import VQModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from matplotlib import pyplot as plt

import random
import utils
import os
import math

from torchvision.utils import make_grid
from drqv2 import RandomShiftsAug

from numpy_replay_buffer import EfficientReplayBuffer
from utils import load_offline_dataset_into_buffer

from dm_env import specs
import dmc

import copy
import wandb
from omegaconf import OmegaConf

from torch.distributions.categorical import Categorical

from network_utils import Predictor, EMA, get_parameter_names, update_moving_average

######################
batch_size = 32

taming_path = "../../" # taming models path
offline_dir = "../../walker_train/vq_latents/" # dataset dir path
vq_model_path = "../../walker_vq_model_step.pth" # vqvae model path
save_dir_path = "./" # voc's gpt model save path

######################

class TFAgent:
    def __init__(self, discount=0.8, augmentation=RandomShiftsAug(pad=4)):

        wandb.init(project="Video Occupancy Models",
               id="voc_vqvae_gamma_{}".format(discount),
               entity=None, dir=os.getcwd())

        self.train_env = dmc.make("offline_walker_walk_expert", 3, 2, 0, None)
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.discount = discount
        self.batch_size = batch_size
        self.nsteps = min(int(1 / (1 - self.discount)), 10) # only sample max 10 steps
        self.codebook_size = 1024

        ######### train on VQ latents #########
        self.replay_buffer = EfficientReplayBuffer(1000000, self.batch_size, 1, self.discount, 3, False, data_specs)
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, None, 3, 1000000, latent_style="vae")
        ##########################

        ######### vq-vae setup #########
        config_path = os.path.join(taming_path, "vqgan_imagenet_f16_1024/configs/model.yaml")
        config = OmegaConf.load(config_path)
        self.model = VQModel(**config.model.params) # Don't put the entire model on device, we only need the decoder
        self.quantizer = self.model.quantize.to('cuda')
        self.decoder = nn.Sequential(self.model.post_quant_conv, self.model.decoder).to('cuda')

        self.from_imagenet = False
        if self.from_imagenet:
            ckpt_path = os.path.join(taming_path, "vqgan_imagenet_f16_1024/ckpts/last.ckpt")
            sd = torch.load(ckpt_path, map_location="cuda")["state_dict"]
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
        else:
            print("Loading fine-tuned VQ model...")
            self.model.load_state_dict(torch.load(vq_model_path))
        ##########################

        ######### gpt setup #########
        configuration = GPT2Config(vocab_size=1024, n_layer=4, n_head=8, n_embed=512, resid_pdrop=0.2, embd_pdrop=0.2, attn_prdrop=0.2)
        self.gpt = GPT2LMHeadModel(configuration).to('cuda')
        self.gpt_target = copy.deepcopy(self.gpt)
        self.gpt_target.generation_config.output_scores = True
        self.target_ema_updater = EMA(0.9)
        ##########################

        ######### optimizer setup #########
        self.reward_predictor = Predictor(256 * 25 * 3).to('cuda')
        self.gpt_optimizer = torch.optim.AdamW(self.get_grouped_params(), lr=3e-4)
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

        self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).to('cuda')
        self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).to('cuda')

        self.device = 'cuda'
        self.aug = augmentation

        self.saving_iter = [50, 100, 500, 1000, 2000, 5000, 10000, 50000, 75000, 100000]
        self.train()

    def get_grouped_params(self):
        decay_parameters = get_parameter_names(self.gpt, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.gpt.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in self.gpt.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def train(self, training=True):
        self.training = training

    def update(self, step=0):
        metrics = dict()
        
        batch, indices = next(self.replay_buffer)
        obs, action, reward, discount, next_obs, latent, _, _, obs_k = utils.to_torch(batch, self.device)

        # collect discrete vq indices
        y_context = obs.long()
        y_target = obs_k.long()
        obs_shape = obs.shape

        quant_target = self.quantizer.get_codebook_entry(y_target, None)
        quant_target = quant_target.view(-1, 5, 5, 256).permute(0, 3, 1, 2)

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
        outputs = self.gpt(inp, labels=inp)
        gpt_loss = outputs.loss
        
        loss = gpt_loss + reward_loss
        
        loss.backward()

        # grad accumulate
        if step % 2 == 0:
            self.optimizer.step()
            self.gpt_optimizer.step()

            self.optimizer.zero_grad()
            self.gpt_optimizer.zero_grad()

            self.lr_scheduler.step()
            update_moving_average(self.target_ema_updater, self.gpt_target, self.gpt)

        # visualize predictions 
        if step % 100 == 0 and step != 0:
            with torch.no_grad():
                # sample a batch of traj and corresponding values
                batch, indices = self.replay_buffer.sample_spr(jumps=self.nsteps)
                _, _, _, _, _, all_obs, all_pixel_obs, _, values = utils.to_torch(batch, self.device)
                
                # preprocess first obs from traj
                obs = all_obs[0]
                obs_shape = obs.shape
                pixel_obs_shape = F.interpolate(all_pixel_obs[0], size=80).shape

                y_context = obs.long()
                
                # sample target predictions
                p_t = self.gpt.generate(y_context, max_new_tokens=y_context.shape[-1], do_sample=True, pad_token_id=-100)[:, -y_context.shape[-1]:]
                
                # reconstruct sampled prediction
                quant = self.quantizer.get_codebook_entry(p_t, None)
                quant = quant.view(-1, 5, 5, 256).permute(0, 3, 1, 2)
                
                p_t_pixel_recon = self.decoder(quant)

                viz_imgs = []
                viz_imgs.append(p_t_pixel_recon)
                for i in range(all_pixel_obs.shape[0]):
                    obs = all_pixel_obs[i]
                    obs = F.interpolate(obs, size=80)
                    obs = torch.stack([torch.einsum('nchw->nhwc', obs[:, i*3:3+i*3] / 255.) - self.imagenet_mean / self.imagenet_std for i in range(3)])
                    obs = torch.einsum('tnhwc->ntchw', obs).reshape((pixel_obs_shape[0] * 3, 3, *pixel_obs_shape[2:])) #torch.einsum('nhwc->nchw', obs)
                    viz_imgs.append(obs)

                value_loss = self.get_value_estimates(y_context, values, obs_shape)
                wandb.log({"value loss": value_loss}, step=step)
                density_value_loss = self.get_density_value_estimates(y_context, all_obs, obs_shape)
                wandb.log({"density value loss": density_value_loss}, step=step)

                viz_imgs = torch.stack(viz_imgs)[:, :8]
                t, n, c, h, w = viz_imgs.shape
                viz_imgs = torch.einsum('tnchw->ntchw', viz_imgs)
                viz_imgs = viz_imgs.reshape(t*n, c, h, w)
                viz_img = make_grid(viz_imgs, nrow=t, normalize=True, scale_each=True)

                img = wandb.Image(viz_img)
                wandb.log({f"Gamma Pred": img}, step=step)

        wandb.log({"reward loss": reward_loss}, step=step)
        wandb.log({"gpt loss": gpt_loss}, step=step)
        
        # save gpt model
        if step in self.saving_iter:
            print("saving gpt weights...")
            # self.save_gpt_weights(step)

        return metrics

    def save_gpt_weights(self, step):
        torch.save(self.gpt.state_dict(), os.path.join(save_dir_path, "vqvae_microgpt_gamma_{}_{}_model_step_{}.pth".format(self.discount, self.codebook_size, step)))

    def get_value_estimates(self, y_context, values, obs_shape):
        # Take a state, get samples from the gamma distribution, 
        # Run the reward predictor through these to get value estimates
        # Get ground truth value estimates by simply taking discounted sum of rewards
        # Compare these for different states
        
        num_gamma_samples = 100
        values_pred = []
        
        for i in range(num_gamma_samples):
            outputs = self.gpt_target.generate(y_context, max_new_tokens=y_context.shape[-1], do_sample=True, output_scores=True, return_dict_in_generate=True, pad_token_id=-100)
            p_t = outputs.sequences[:, -y_context.shape[-1]:]
            
            quant = self.quantizer.get_codebook_entry(p_t, None)
            quant = quant.view(-1, 5, 5, 256).permute(0, 3, 1, 2)
            
            values_pred.append(self.reward_predictor(quant.float().reshape(obs_shape[0], -1)).squeeze(1))

        values_pred = torch.stack(values_pred).sum(0) / (100 * (1 - self.discount))

        value_estimation_loss = F.mse_loss(values_pred, values.squeeze(1).float()).mean()
        print("val estimation", value_estimation_loss, values_pred[:5], values[:5])
        
        return value_estimation_loss

    def get_density_value_estimates(self, y_context, all_obs, obs_shape):

        values_pred = []
        values_actual = []
        for i in range(all_obs.shape[0]-1):
            y_target = all_obs[i+1].long()
            quant_target = self.quantizer.get_codebook_entry(y_target, None)
            quant_target = quant_target.view(-1, 5, 5, 256).permute(0, 3, 1, 2)
        
            inp = torch.cat([y_context, y_target], dim=1)
            outputs = self.gpt(inp, labels=inp)
            logits = outputs.logits[:, -y_target.shape[1]-1:-1]
            scores = torch.nn.functional.log_softmax(logits, dim=2)

            gathered_scores = torch.gather(scores, dim=2, index=y_target.unsqueeze(2))
            gathered_logits = torch.gather(logits, dim=2, index=y_target.unsqueeze(2))

            input_length = y_target.shape[1]
            output_length = input_length + torch.sum(gathered_logits < 0, dim=1)
            prob = torch.exp(gathered_scores.sum(1) / output_length)
            values_pred.append(prob.squeeze(1) * self.reward_predictor(quant_target.float().reshape(obs_shape[0], -1)).squeeze(1))
            values_actual.append(self.reward_predictor(quant_target.float().reshape(obs_shape[0], -1)).squeeze(1))

        values_pred = torch.stack(values_pred).sum(0)

        discount_vec = torch.pow(self.discount, torch.arange(torch.stack(values_actual).shape[0], device='cuda'))
        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        values_actual = torch.sum(torch.stack(values_actual) * discount_vec.repeat(torch.stack(values_actual).shape[1], 1).T, dim=0)

        value_estimation_loss = F.mse_loss(values_pred, values_actual).mean()
        print("density val estimation", value_estimation_loss, values_pred[:5], values_actual[:5])

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