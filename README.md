## Video Occupancy Models

Code for the paper `Video Occupancy Models`, includes three versions of quantizing the input video frames -- `vae` which uses a VQ-VAE, `dino` which uses quantized DINO, and `musik` which uses quantized Multi-step Inverse Dynamics. 

<img width="847" alt="Screenshot 2024-07-16 at 12 05 30 PM" src="https://github.com/user-attachments/assets/cbcfbf89-b5aa-42b1-86ba-8c4f89e41276">

This is a PyTorch/GPU implementation of the paper [Video Occupancy Models](https://arxiv.org/pdf/2407.09533):
```
@Article{VideoOccupancyModels2024,
  author  = {Manan Tomar and Philippe Hansen-Estruch and Philip Bachman and Alex Lamb and John Langford and Matthew E. Taylor and Sergey Levine,
  journal = {arXiv:2407.09533},
  title   = {Video Occupancy Models},
  year    = {2024},
}
```
### Installation

The main packages are provided in the `requirements.txt` file. This code has been tested on a virtual env with Python-3.8 with the package versions listed in the requirements file.

### Model Checkpoints and Datasets

The following table provides the pre-trained model checkpoints and datasets used in the paper:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Cheetah</th>
<th valign="bottom">Walker</th>
<!-- TABLE BODY -->
<tr><td align="left">VQ-VAE fine-tuned model checkpoint</td>
<td align="center"><a href="https://huggingface.co/manantomar/video-occupancy-models/resolve/main/cheetah_vq_vae_model.pth?download=true">download</a></td>
<td align="center"><a href="https://huggingface.co/manantomar/video-occupancy-models/resolve/main/walker_vq_vae_model.pth?download=true">download</a></td>
</tr>
<tr><td align="left">DINO latent datasets</td>
<td align="center"><a href="https://huggingface.co/datasets/manantomar/video-occupancy-models-datasets/tree/main/cheetah_train/dino_latents">link</a></td>
</tr>
<tr><td align="left">VQ-VAE latent datasets</td>
<td align="center"><a href="https://huggingface.co/datasets/manantomar/video-occupancy-models-datasets/tree/main/cheetah_train/vq_latents">link</a></td>
<td align="center"><a href="https://huggingface.co/datasets/manantomar/video-occupancy-models-datasets/tree/main/walker_train/vq_latents">link</a></td>
</tr>
</tbody></table>

### VQ-VAE VOC

You would need to download the contents of this [folder](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/) and place them one directory above where this repo is present. This folder contains model descriptions for using a VQ-VAE model from the [taming-transformers](https://github.com/CompVis/taming-transformers?tab=readme-ov-file) codebase.

Run [train_vq_vae_voc.py](https://github.com/manantomar/video-occupancy-models/blob/master/vae/train_vq_vae_voc.py) to train a VOC model on stored VQ-VAE latents. If you want to train both the VQ-VAE and the VOC model on pixel data then run [train_pixel_vq_vae_voc.py](https://github.com/manantomar/video-occupancy-models/blob/master/vae/train_pixel_vq_vae_voc.py). In case you want to create your own latents by traning VQ-VAE on a custom dataset use the `collect_latents()` and `train_vq_latents()` methods in [save_vq_codes.py](https://github.com/manantomar/video-occupancy-models/blob/master/vae/save_vq_codes.py).

### DINO VOC

We use a quantized verison of [DINO](https://arxiv.org/abs/2104.14294) from [BEiT-v2](https://github.com/microsoft/unilm/tree/master/beit2). You would need to download this [dino model file](https://github.com/addf400/files/releases/download/BEiT-v2/vqkd_encoder_base_decoder_1x768x12_dino-663c55d7.pth) and place them one directory above where this repo is present.

Run [train_vq_dino_voc.py](https://github.com/manantomar/video-occupancy-models/blob/master/dino/train_vq_dino_voc.py) to train a VOC model on stored DINO latents. Again, in case you want to create your own latents by running a quantized version of DINO on a custom dataset use the `collect_latents()` method in [save_dino_codes.py](https://github.com/manantomar/video-occupancy-models/blob/master/dino/save_dino_codes.py).

### MUSIK VOC

In the case, action data is also available, we use a quantized multi-step inverse kinematics (MUSIK) objective to train the representation.

Run [train_vq_musik_voc.py](https://github.com/manantomar/video-occupancy-models/blob/master/musik/train_vq_musik_voc.py) to train a VOC model along with the [MUSIK](https://arxiv.org/pdf/2211.00164) objective on pixel data. 
