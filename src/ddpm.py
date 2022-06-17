import copy
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from src.dataset.get_data import get_train_data
from src.diffusion import GaussianDiffusion, make_beta_schedule
from src.torch_utils import set_requires_grad, trainable_parameters


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class DDPM(pl.LightningModule):
    def __init__(self, conf: DictConfig, save_hyperparameters=False):
        super().__init__()

        self.conf = conf
        if save_hyperparameters:
            self.save_hyperparameters(ignore="save_hyperparameters")

        from .unet import Model
        self.model = Model(
            in_channels=self.conf.model.in_channels,
            out_channels=self.conf.model.out_channels,
            ch=self.conf.model.channels,
            ch_mult=tuple(self.conf.model.channel_multiplier),
            num_res_blocks=self.conf.model.n_res_blocks,
            resolution=self.conf.dataset.resolution,
            attn_resolutions=[self.conf.dataset.resolution // stride for stride in self.conf.model.attn_strides],
            dropout=self.conf.model.dropout
        )

        self.ema = copy.deepcopy(self.model)
        set_requires_grad(self.ema, False)

        self.betas = make_beta_schedule(schedule=self.conf.model.schedule)

        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type=self.conf.model.mean_type,
            model_var_type=self.conf.model.var_type,
            is_implicit=self.conf.model.is_implicit,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        data_dir = self.conf.dataset.root
        self.train_set, self.valid_set = get_train_data(self.conf, data_dir=data_dir)

    def configure_optimizers(self):
        if self.conf.training.optimizer.type == "adam":
            wd = self.conf.training.optimizer.wd
            optimizer = torch.optim.Adam(
                trainable_parameters(self.model),
                lr=self.conf.training.optimizer.lr,
                weight_decay=wd,
            )
        else:
            raise NotImplementedError()

        return optimizer

    def train_dataloader(self):
        loader_conf = self.conf.training.dataloader
        train_loader = DataLoader(
            self.train_set,
            batch_size=loader_conf.batch_size,
            num_workers=loader_conf.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

        return train_loader

    def val_dataloader(self):
        loader_conf = self.conf.validation.dataloader
        valid_loader = DataLoader(
            self.valid_set,
            batch_size=loader_conf.batch_size,
            num_workers=loader_conf.num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

        return valid_loader

    def on_train_epoch_start(self) -> None:
        if self.current_epoch <= self.conf.training.ema.start_epoch:
            if self.current_epoch == self.conf.training.ema.start_epoch:
                print(f"\n-----------EMA START --------\n")
            accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                       decay=0)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: int = 0) -> None:
        if self.current_epoch >= self.conf.training.ema.start_epoch:
            accumulate(self.ema, self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                       decay=self.conf.training.ema.decay)

    def forward(self, x):
        return self.diffusion.p_sample_loop(self.model, x.shape, device=x.device)

    def _step(self, model, batch):
        img, _ = batch
        t = torch.randint(0, self.diffusion.num_timesteps, size=(img.shape[0],), device=img.device)

        losses, loss_dict = self.diffusion.training_losses(model, img, t)

        loss = losses.mean()

        return loss, loss_dict

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, loss_dict = self._step(self.model, batch)

        log_dict = {"train/loss": loss.item()}
        for k, v in loss_dict.items():
            log_dict[f"train/{k}"] = v.mean().item()
        self.log_dict(log_dict, prog_bar=False, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, loss_dict = self._step(self.ema, batch)

        log_dict = {"val/loss": loss.item()}
        for k, v in loss_dict.items():
            log_dict[f"val/{k}"] = v.mean().item()
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
