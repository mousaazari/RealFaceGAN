from encoder import DualSpaceEncoder
from psp_testing_options import TestOptions
import os
import math
from utils import lpips
from torch.utils.data import DataLoader
from utils.dataset_projector import MultiResolutionDataset
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


def make_noise(log_size=8):
    device = 'cuda'

    noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
    return noises


def noise_regularize_(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * torch.unsqueeze(strength, -1)

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )


def editing(z_code, p_code):
    return z_code, p_code


if __name__ == '__main__':
    device = 'cuda'

    psp_config = TestOptions()
    psp_config.parser.add_argument('--lr_rampup', type=float, default=0.05)
    psp_config.parser.add_argument('--lr_rampdown', type=float, default=0.25)
    psp_config.parser.add_argument('--lr', type=float, default=0.1)
    psp_config.parser.add_argument('--noise', type=float, default=0.05)
    psp_config.parser.add_argument('--noise_ramp', type=float, default=0.75)
    psp_config.parser.add_argument('--noise_regularize', type=float, default=1e5)
    psp_config.parser.add_argument('--mse', type=float, default=0)
    psp_config.parser.add_argument('--encode_batch', type=int, default=8)
    psp_config.parser.add_argument('--optimization_batch', type=int, default=1)
    psp_config.parser.add_argument('--seed', type=int, default=1)
    psp_config.parser.add_argument('--dataset_dir', type=str,
                                   required=True)
    psp_config.parser.add_argument('--loop', type=int, default=2000)

    args = psp_config.parse()

    args.output_dir = os.path.join(args.output_dir, "encoder_inversion", f"{args.dataset_type}")

    args.latent = 512
    args.token = 2 * (int(math.log(args.size, 2)) - 1)

    args.use_spatial_mapping = not args.no_spatial_map

    os.makedirs(args.output_dir, exist_ok=True)

    args.encoded_z_npy = os.path.join(args.output_dir, "encoded_z.npy")
    args.encoded_p_npy = os.path.join(args.output_dir, "encoded_p.npy")

    network = DualSpaceEncoder(args)

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    res_latent = []
    res_param = []
    res_perceptual_values = []
    res_mse_values = []

    dataset = MultiResolutionDataset(args.dataset_dir, resolution=args.size)

    if os.path.isfile(args.encoded_z_npy):
        print("encoded_npy exists!")
        z_latent_codes = np.load(args.encoded_z_npy)
        p_latent_codes = np.load(args.encoded_p_npy)
    else:
        loader = DataLoader(dataset, shuffle=False, batch_size=args.encode_batch, num_workers=4, drop_last=False)

        z_latent_codes_enc = []
        p_latent_codes_enc = []

        with tqdm(desc='Generation', unit='it', total=len(loader)) as pbar_1:
            for it, images in enumerate(iter(loader)):
                images = images.to(device)
                z_code, p_code = network.encode(images)
                z_latent_codes_enc.append(z_code.cpu().detach().numpy())
                p_latent_codes_enc.append(p_code.cpu().detach().numpy())

                pbar_1.update()

            z_latent_codes = np.concatenate(z_latent_codes_enc, axis=0)
            p_latent_codes = np.concatenate(p_latent_codes_enc, axis=0)
            np.save(f"{args.encoded_z_npy}", z_latent_codes)
            np.save(f"{args.encoded_p_npy}", p_latent_codes)

    print('z_latent_codes.shape', z_latent_codes.shape)
    print('p_latent_codes.shape', p_latent_codes.shape)
