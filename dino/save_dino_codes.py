import sys
sys.path.insert(0, '../utils/')

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

from torchvision import transforms as pth_transforms
from timm.models import create_model

import modeling_vqkd 

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

class DINOAgent:
    def __init__(self, augmentation=RandomShiftsAug(pad=4)):

        wandb.init(project="Video Occupancy Models",
               entity=None, dir=os.getcwd())

        self.hdf5_file_path = "../../cheetah_train/org/3_cheetah_run_random.hdf5"
        self.save_path = "../../cheetah_train/dino_latents/3_random_dino_latents.hdf5"
        self.dino_path = '../../vqkd_encoder_base_decoder_1x768x12_dino-663c55d7.pth'

        self.model = create_model(
            'vqkd_encoder_base_decoder_1x768x12_dino',
            pretrained=True,
            pretrained_weight=self.dino_path,
            as_tokenzer=True,
            ).to('cuda').eval()

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
        obs = F.interpolate(obs, size=224)

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
        
        dino_latents = []
        for i in range(int(iters)):
            obs = fobs[i*batch_size:(i + 1)*batch_size]
            obs = utils.to_torch([obs], self.device)[0]

            obs, obs_shape = self.process_obs(obs)
            
            # dino embed
            quant_context, y_context, context_loss = self.model.encode(obs)
        
            y_context = y_context.reshape((obs_shape[0], -1)).detach()
        
            # collect discrete vq indices
            dino_latents.append(y_context)

        dataset_names = ['action', 'discount', 'observation', 'reward', 'step_type', 'dino_latents']
        data_arrays = [faction, fdiscount, fobs, freward, fstep_type, torch.stack(dino_latents).reshape((batch_size * int(iters), -1)).cpu().long()]
        self.create_hdf5_file(self.save_path, dataset_names, data_arrays)

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


if __name__ == "__main__":

    agent = DINOAgent()
    agent.collect_latents()
