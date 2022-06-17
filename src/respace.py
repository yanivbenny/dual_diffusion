from copy import deepcopy

import torch

from .diffusion import GaussianDiffusion


def respace_timesteps(old_num_timesteps, new_num_timesteps, mode):
    assert mode in ['ddodm', 'ddim']
    if mode == 'ddodm':
        return list(reversed(range(old_num_timesteps - 1, 0, -old_num_timesteps // new_num_timesteps)))
    elif mode == 'ddim':
        return list(range(0, old_num_timesteps, old_num_timesteps // new_num_timesteps))


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, base_diffusion, use_timesteps, **kwargs):

        use_timesteps = set(use_timesteps)
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        default_kwargs = {
            "model_mean_type": base_diffusion.model_mean_type,
            "model_var_type": base_diffusion.model_var_type,
            "is_implicit": base_diffusion.is_implicit,
        }
        default_kwargs.update(kwargs)
        default_kwargs["betas"] = torch.tensor(new_betas, dtype=base_diffusion.betas.dtype)
        super().__init__(**default_kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map)


class _WrappedModel:
    def __init__(self, model, timestep_map):
        self.model = model
        self.timestep_map = timestep_map

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)
