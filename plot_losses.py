import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from src.ddpm import DDPM
from src.torch_utils import to_cpu

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.")
    parser.add_argument('--model_paths', type=str, nargs='+', help="A list of model checkpoints to use")

    # Parse arguments
    args, argv = parser.parse_known_args()

    # Load config
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)
    OmegaConf.register_new_resolver("out_channels", lambda x: 7 if x == "dualx" else 3)
    conf = OmegaConf.create(conf, flags={'struct': True})

    # Update config with parser
    argv_conf = OmegaConf.from_dotlist(argv)
    conf = OmegaConf.merge(conf, argv_conf)

    # Load model
    ddpms = []
    for model_path in args.model_paths:
        ddpm = DDPM.load_from_checkpoint(model_path, save_hyperparameters=False)
        ddpm.eval()
        ddpm.cuda()
        ddpms.append(ddpm)

    ddpms[0].setup('validate')
    val_loader = ddpms[0].val_dataloader()

    # Collect losses
    T = conf.model.schedule.n_timestep
    dT = 50

    losses = [{} for _ in range(len(ddpms))]
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, targets = batch
            images = images.cuda()
            for ti, t in enumerate(range(dT-1, T, dT)):
                t = torch.LongTensor(size=(images.shape[0],)).fill_(t).cuda()

                for i, ddpm in enumerate(ddpms):
                    _, loss_dict = ddpm.diffusion.training_losses(ddpm.ema, images, t)

                    for k, v in loss_dict.items():
                        losses[i].setdefault(k, [[] for _ in range(T // dT)])
                        losses[i][k][ti].append(to_cpu(v))

    print('Done')
    legend = [dd.conf.model.mean_type for dd in ddpms]
    for loss_key in losses[0].keys():
        plt.figure()
        for i in range(len(ddpms)):
            y = [torch.cat(loss_t, dim=0) for loss_t in losses[i][loss_key]]
            yt_mean = np.array([yt.mean().item() for yt in y])
            yt_err = np.array([yt.var().sqrt().item() for yt in y])
            z = plt.plot(range(dT, T+1, dT), yt_mean, label=legend[i])

            plt.fill_between(range(dT, T+1, dT), yt_mean - yt_err, yt_mean + yt_err,
                            alpha=0.2, color=z[0].get_color())

        plt.legend()
        plt.title(loss_key)
        plt.semilogy()
        plt.xlim(dT, T)
        plt.show()
