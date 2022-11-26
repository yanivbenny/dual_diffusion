import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.visualization import VisualizationCallback
from src.ddpm import DDPM
from src.helpers import str2bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/cifar10_train.yaml", help="Path to config.")
    parser.add_argument("--save_dir", type=str, default="./experiments")

    # Training specific args
    parser.add_argument("--ckpt_freq", type=int, default=10, help="Frequency of saving the model (in epoch).")
    parser.add_argument("--vis_num", type=int, default=16, help="Number of samples to visualize.")
    parser.add_argument("--num_vis_timesteps", type=int, default=100, help="Num timesteps in respaced diffusion during visualization.")

    parser.add_argument("--n_gpu", type=int, default=1, help="Number of available GPUs.")
    parser.add_argument("--precision", type=int, default=32, help="Number of bits")

    # Weights & Biases args
    parser.add_argument("--wb_entity", type=str, default=None)
    parser.add_argument("--wb_project", type=str, default="dual_ddpm_{dataset}_{resolution}")
    parser.add_argument("--wb_name", type=str, default=None)

    # Debug args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--debug", type=str2bool, default=False)

    # Parse arguments
    args, argv = parser.parse_known_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load config
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    OmegaConf.register_new_resolver("out_channels", lambda x: 7 if x == "dualx" else 3)

    conf = OmegaConf.create(conf, flags={"struct": True})
    OmegaConf.resolve(conf)

    # Update config with parser
    argv_conf = OmegaConf.from_dotlist(argv)
    conf = OmegaConf.merge(conf, argv_conf)

    # Format arguments
    args.wb_project = args.wb_project.format(dataset=conf.dataset.name, resolution=conf.dataset.resolution)

    # Initialize logger
    os.makedirs(os.path.join(args.save_dir, "wandb"), exist_ok=True)
    logger = WandbLogger(
        entity=args.wb_entity,
        project=args.wb_project,
        name=args.wb_name,
        save_dir=args.save_dir,
    )

    # Load model
    ddpm = DDPM(conf, save_hyperparameters=True)

    # Initialize callbacks
    callbacks = []
    if args.ckpt_freq > 0:
        callbacks.append(
            VisualizationCallback(
                every_n_epochs=args.ckpt_freq,
                vis_num=args.vis_num,
                num_vis_timesteps=args.num_vis_timesteps,
            )
        )
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.ckpt_freq,
                monitor="val/loss",
                mode="min",
                verbose=True,
                save_last=True,
                save_top_k=1,
                save_weights_only=False,
            )
        )

    # Initialize trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=args.num_sanity_val_steps,
        default_root_dir=args.save_dir,
        logger=logger,
        gpus=args.n_gpu,
        strategy="ddp" if args.n_gpu != 1 else None,
        # debug
        max_steps=          -1    if args.debug else conf.training.n_iter,
        max_epochs=         5     if args.debug else conf.training.n_epochs,
        limit_train_batches=100   if args.debug else conf.training.get("limit_train_batches", 1.0),
        limit_val_batches=  100   if args.debug else conf.training.get("limit_val_batches", 1.0),
        #
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=1.,
        accumulate_grad_batches=conf.training.accumulate_grad_batches,
    )

    trainer.fit(ddpm)


if __name__ == "__main__":
    main()
    print("Done")
