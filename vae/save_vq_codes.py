import sys
sys.path.insert(0, '../utils/')

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
import wandb
import math

from torchvision.utils import make_grid
from drqv2 import RandomShiftsAug

from dm_env import specs
import dmc

import h5py
from omegaconf import OmegaConf

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

class VQAgent:
    def __init__(self, augmentation=RandomShiftsAug(pad=4)):

        wandb.init(project="Video Occupancy Models",
               entity=None, dir=os.getcwd())

        self.hdf5_dir_path = "../../walker_train/org/"
        self.hdf5_file_path = "../../walker_train/org/3_walker_walk_medium.hdf5"
        self.save_path = "../../walker_train/vq_latents/3_medium_vq_latents.hdf5"
        self.taming_path = "../../"
        self.vq_model_path = "../../walker_vq_model.pth"

        ################################################################################
        #                                                                              #
        #                                VQ VAE Setup                                  #
        #                                                                              #
        ################################################################################
        config_path = os.path.join(self.taming_path, "vqgan_imagenet_f16_1024/configs/model.yaml")
        config = OmegaConf.load(config_path)

        self.model = VQModel(**config.model.params).to('cuda')

        self.from_imagenet = False
        if self.from_imagenet:
            ckpt_path = os.path.join(self.taming_path, "vqgan_imagenet_f16_1024/ckpts/last.ckpt")
            sd = torch.load(ckpt_path, map_location="cuda")["state_dict"]
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
        else:
            print("Loading fine-tuned VQ model...")
            self.model.load_state_dict(torch.load(self.vq_model_path))

        # vq vae optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)

        self.imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).to('cuda')
        self.imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).to('cuda')

        self.device = 'cuda'
        # data augmentation
        self.aug = augmentation

        self.train()

    def train(self, training=True):
        self.training = training

    def process_obs(self, obs):
        obs = self.aug(obs.float())
        obs = F.interpolate(obs, size=80)

        obs_shape = obs.shape
        
        obs = torch.einsum('nchw->nhwc', obs / 255.) - self.imagenet_mean / self.imagenet_std
        obs = torch.einsum('nhwc->nchw', obs).reshape((obs_shape[0], 3, *obs_shape[2:]))
        return obs, obs_shape

    def collect_latents(self, step=0):
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            fobs = hf['observation'][()]
            faction = hf['action'][()]
            fdiscount = hf['discount'][()]
            freward = hf['reward'][()]
            fstep_type = hf['step_type'][()]

        fobs = fobs
        batch_size = 25
        
        assert fobs.shape[0] % batch_size == 0
        iters = fobs.shape[0] / batch_size

        print("Total Obs are {}, Batch Size is {}, Total Iterations is {}".format(fobs.shape[0], batch_size, iters))
        
        vae_latents = []
        for i in range(int(iters)):
            obs = fobs[i*batch_size:(i + 1)*batch_size]
            obs = utils.to_torch([obs], self.device)[0]

            obs, obs_shape = self.process_obs(obs)
            
            # vq embed
            quant_context, emb_loss_context, info_context = self.model.encode(obs)
            
            # collect discrete vq indices
            y_context = info_context[2].view(obs_shape[0], -1).detach()
            vae_latents.append(y_context)

        dataset_names = ['action', 'discount', 'observation', 'reward', 'step_type', 'vae_latents']
        data_arrays = [faction, fdiscount, fobs, freward, fstep_type, torch.stack(vae_latents).reshape((batch_size * int(iters), -1)).cpu().long()]
        self.create_hdf5_file(self.save_path, dataset_names, data_arrays)

    def train_vq_latents(self):
        hdf5_files = [f for f in os.listdir(self.hdf5_dir_path) if f.endswith('.hdf5')]

        fobs = []
        # Loop through each file and read data
        for file in hdf5_files:
            file_path = os.path.join(self.hdf5_dir_path, file)
            with h5py.File(file_path, 'r') as hf:
                fobs.append(hf['observation'][()])
        fobs = np.stack(fobs)
        fobs = fobs.reshape((-1, *fobs.shape[2:]))

        batch_size = 64
        
        vae_latents = []
        
        for step in range(50000): # Num of updates
            idx = np.random.choice(fobs.shape[0], size=batch_size)
            obs = fobs[idx]
            obs = utils.to_torch([obs], self.device)[0]

            obs, obs_shape = self.process_obs(obs)

            # vq embed
            quant_context, emb_loss_context, info_context = self.model.encode(obs)
            
            xrec, qloss = self.model.decode(quant_context), emb_loss_context
            vae_loss, log_dict_ae = self.model.loss(qloss, obs, xrec, 0, step, last_layer=self.model.get_last_layer(), split="train")

            # collect discrete vq indices
            y_context = info_context[2].view(obs_shape[0], -1).detach()

            if step % 100 == 0:
                with torch.no_grad():
                    print("vae loss", vae_loss)
                    viz_imgs = []
                    viz_imgs.append(xrec)
                    viz_imgs.append(obs)

                    viz_imgs = torch.stack(viz_imgs)[:, :5]
                    t, n, c, h, w = viz_imgs.shape
                    viz_imgs = torch.einsum('tnchw->ntchw', viz_imgs)
                    viz_imgs = viz_imgs.reshape(t*n, c, h, w)
                    viz_img = make_grid(viz_imgs, nrow=t, normalize=True, scale_each=True)

                    img = wandb.Image(viz_img)
                    wandb.log({f"Gamma Pred": img}, step=step)

            loss = vae_loss
            loss.backward()

            if step % 1 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if step % 2000 == 0:
                self.save_vq_weights(step)

    def create_hdf5_file(self, file_path, dataset_names, data_arrays):
        """
        Create an HDF5 file and store multiple arrays in it.

        Parameters:
        - file_path: Path to the HDF5 file.
        - dataset_names: List of dataset names.
        - data_arrays: List of NumPy arrays to be stored in the HDF5 file.
        """
        # org dataset ['action', 'discount', 'observation', 'reward', 'step_type']
        with h5py.File(file_path, 'w') as hf:
            for dataset_name, data_array in zip(dataset_names, data_arrays):
                # Create a dataset in the HDF5 file
                hf.create_dataset(dataset_name, data=data_array)

    def save_vq_weights(self, step):
        torch.save(self.model.state_dict(), self.vq_model_path)


if __name__ == "__main__":

    agent = VQAgent()
    agent.collect_latents()

    # agent.train_vq_latents()
