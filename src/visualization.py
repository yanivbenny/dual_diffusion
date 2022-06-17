import time

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid

from src.helpers import exists
from src.respace import SpacedDiffusion, respace_timesteps


class VisualizationCallback(pl.Callback):
    def __init__(self, every_n_epochs=1, vis_num=16, num_vis_timesteps=None, is_implicit=True):
        super().__init__()
        self.vis_freq = every_n_epochs
        self.vis_num = vis_num
        self.num_vis_timesteps = num_vis_timesteps
        self.is_implicit = is_implicit

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._define_metrics(trainer)

    def _define_metrics(self, trainer: "pl.Trainer"):
        run = trainer.logger.experiment
        run.define_metric("trainer/epoch")

        run.define_metric("generated_images",               step_metric="trainer/epoch", step_sync=True)
        run.define_metric("progressive_generated_images",   step_metric="trainer/epoch", step_sync=True)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Compute visualization during sanity check and after every K epochs
        # During sanity check only compute visualization but don't log
        if not trainer.sanity_checking and (trainer.current_epoch+1) % self.vis_freq != 0:
            return

        t_start = time.time()

        # Possibly respace diffusion schedule (generate with fewer steps)
        diffusion = pl_module.diffusion
        was_is_implicit = diffusion.is_implicit
        if exists(self.num_vis_timesteps):
            use_timesteps = respace_timesteps(
                old_num_timesteps=diffusion.num_timesteps,
                new_num_timesteps=self.num_vis_timesteps,
                mode='ddodm',
            )
            diffusion = SpacedDiffusion(diffusion, use_timesteps=use_timesteps)
            diffusion.to(pl_module.device)
        diffusion.is_implicit = self.is_implicit

        # Generation
        self._log_generation(trainer=trainer, pl_module=pl_module, diffusion=diffusion)

        diffusion.is_implicit = was_is_implicit

        t_end = time.time()
        print(f'Visualization duration in minutes:seconds - {int(t_end - t_start) // 60}:{int(t_end - t_start) % 60}')

    def _log_generation(self, trainer, pl_module, diffusion):
        device = pl_module.device

        shape = (self.vis_num, 3, pl_module.conf.dataset.resolution, pl_module.conf.dataset.resolution)

        samples, progressive_samples = progressive_samples_fn(pl_module.ema, diffusion, shape,
                                        include_x0_pred_timesteps=diffusion.num_timesteps // 20,
                                        device=device, verbose=True)

        wandb_logs = {'trainer/epoch': trainer.current_epoch}

        imgs = samples.clip(0, 1)
        grid = make_grid(imgs, nrow=4)
        wandb_logs['generated_images'] = wandb.Image(grid)

        imgs = progressive_samples.clip(0, 1)
        grid = make_grid(imgs.reshape(-1, 3, pl_module.conf.dataset.resolution, pl_module.conf.dataset.resolution), nrow=imgs.shape[1])
        wandb_logs['progressive_generated_images'] = wandb.Image(grid)

        # Log
        if not trainer.sanity_checking:
            trainer.logger.experiment.log(wandb_logs, commit=False)


def progressive_samples_fn(
        model,
        diffusion,
        shape,
        device,
        noise_fn=torch.randn,
        include_x0_pred_timesteps=50,
        clip_denoised=True,
        verbose=False,
):

    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=noise_fn,
        device=device,
        include_x0_pred_timesteps=include_x0_pred_timesteps,
        clip_denoised=clip_denoised,
        verbose=verbose
    )

    return (samples + 1)/2, (progressive_samples + 1)/2
