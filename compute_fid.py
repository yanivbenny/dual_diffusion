import argparse
import math
from copy import deepcopy

import torch
from tqdm import tqdm

from src.ddpm import DDPM
from src.fid import EnhancedFID
from src.helpers import str2bool
from src.respace import SpacedDiffusion, respace_timesteps


def samples_fn(
        model,
        diffusion,
        shape,
        device,
        noise_fn=torch.randn,
        clip_denoised=True,
        verbose=False,
        normalize=True,
):

    samples = diffusion.p_sample_loop(
        model=model,
        shape=shape,
        device=device,
        noise_fn=noise_fn,
        clip_denoised=clip_denoised,
        verbose=verbose
    )

    if normalize:
        samples = (samples + 1)/2
    return samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default='val', choices=['train', 'val'])
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=math.inf, help="maximum number of batches to run")
    parser.add_argument("--compute_freq", type=int, default=math.inf, help="compute intermediate FID after every N batches")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--use_ema", type=int, default=True)
    parser.add_argument("--implicit", type=str2bool, default=True)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--respace_mode", type=str, default="ddim", choices=["ddim", "ddodm"])
    parser.add_argument("--clip_denoised", type=str2bool, default=True)

    # Parse arguments
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    ddpm = DDPM.load_from_checkpoint(args.model_path, save_hyperparameters=False)

    if args.num_timesteps:
        old_diffusion = deepcopy(ddpm.diffusion)
        old_timesteps = old_diffusion.num_timesteps

        use_timesteps = respace_timesteps(old_timesteps, args.num_timesteps, mode=args.respace_mode)
        spaced_diffusion = SpacedDiffusion(
            ddpm.diffusion,
            use_timesteps=use_timesteps,
            is_implicit=args.implicit,
        )

        ddpm.diffusion = spaced_diffusion

    mean_type_str = ddpm.diffusion.model_mean_type
    var_type_str = ddpm.diffusion.model_var_type if not args.implicit else 'implicit'

    ddpm.eval()
    ddpm.cuda()

    ddpm.setup()
    if args.split == 'train':
        ddpm.conf.training.dataloader.batch_size = args.batch_size
        loader = ddpm.train_dataloader()
    else:
        ddpm.conf.validation.dataloader.batch_size = args.batch_size
        loader = ddpm.val_dataloader()

    fid_metric = EnhancedFID()
    fid_metric.cuda()

    losses = {}
    with torch.no_grad():
        for b, batch in enumerate(tqdm(loader)):
            if b >= args.max_batches:
                break
            images, targets = batch
            images = images.cuda()

            fake_images = samples_fn(
                model=ddpm.ema if args.use_ema else ddpm.model,
                diffusion=ddpm.diffusion,
                shape=images.shape,
                device=ddpm.device,
                clip_denoised=args.clip_denoised,
                verbose=False,
                normalize=False,
            )
            fake_images.clip_(-1, 1)

            assert (-1 <= images.min() and images.min() < -0.5) and (0.5 <= images.max() and images.max() <= 1), f'({images.min()}, {images.max()})'
            assert (-1 <= fake_images.min() and fake_images.min() < -0.5) and (0.5 <= fake_images.max() and fake_images.max() <= 1), f'({fake_images.min()}, {fake_images.max()})'

            if b == 0:
                print('real')
                print(images.mean(), images.var())
                print(images.mean(dim=(0,2,3)), images.var(dim=(0,2,3)))
                print('fake')
                print(fake_images.mean(), fake_images.var())
                print(fake_images.mean(dim=(0,2,3)), fake_images.var(dim=(0,2,3)))

            fid_metric.update(images.add(1).div_(2).mul_(255).round_().byte(), real=True)
            fid_metric.update(fake_images.add(1).div_(2).mul_(255).round_().byte(), real=False)

            if (b+1) % args.compute_freq == 0 and (b+1) != len(loader):
                fid_dict = fid_metric.compute()

                print(f'{b+1} - {mean_type_str} {var_type_str}: {fid_dict["FID"]:0.4f}')

    fid_dict = fid_metric.compute()

    print(f'{mean_type_str} {var_type_str}: {fid_dict["FID"]:0.4f}')

    print('Done')
