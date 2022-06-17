import math

import torch
from torch import nn

from src.helpers import default

from tqdm import tqdm

from omegaconf.dictconfig import DictConfig


def make_beta_schedule(schedule: "DictConfig"):

    def _cosine_beta(n_timestep, temp1=1, temp2=2, max_clip=0.999, s=0.008):
        t = torch.linspace(0, 1, n_timestep + 1, dtype=torch.float64)
        f = torch.cos(((t + s) / (1 + s)) ** temp1 * math.pi / 2) ** temp2
        alphas_bar = f / f[0]
        betas = (1 - alphas_bar[1:] / alphas_bar[:-1]).clip(max=max_clip)
        return betas

    if schedule.type == "const":
        betas = schedule.beta_end * torch.ones(schedule.n_timestep, dtype=torch.float64)
    elif schedule.type == "quad":
        betas = torch.linspace(schedule.beta_start ** 0.5, schedule.beta_end ** 0.5, schedule.n_timestep, dtype=torch.float64) ** 2
    elif schedule.type == "linear":
        betas = torch.linspace(schedule.beta_start, schedule.beta_end, schedule.n_timestep, dtype=torch.float64)
    elif schedule.type == "cosine":
        betas = _cosine_beta(schedule.n_timestep, temp1=schedule.cosine_temp1, temp2=schedule.cosine_temp2)
    else:
        raise NotImplementedError(schedule)

    return betas


def extract(input, t, shape):
    out     = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out     = out.reshape(*reshape)

    return out


class GaussianDiffusion(nn.Module):
    def __init__(self, betas, model_mean_type, model_var_type, is_implicit=False):
        super().__init__()

        betas = betas.type(torch.float64)

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.is_implicit = is_implicit

        self.register_schedule(betas)

    def register_schedule(self, betas):
        timesteps          = betas.shape[0]
        self.num_timesteps = int(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_alphas_cumprod_prev", torch.sqrt(alphas_cumprod_prev))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod_prev", torch.sqrt(1. - alphas_cumprod_prev))
        self.register("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped",
                      torch.log(torch.cat([posterior_variance[1].view(1,), posterior_variance[1:]])))
        self.register("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register("posterior_mean_coef2", ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)))

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def model_var_log_var(self, model_var_type):
        if model_var_type == 'fixedlarge':
            var = self.betas
            log_var = torch.log(torch.cat([self.betas[1].view(1,), self.betas[1:].view(-1,)]))
        elif model_var_type == 'fixedsmall':
            var = self.posterior_variance
            log_var = self.posterior_log_variance_clipped

        return var, log_var

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def q_posterior_mean(self, x_0, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        return mean

    def q_posterior_mean_implicit(self, x_0, x_t, t, noise):
        if x_0 is None:
            # predict mean using x_t and noise
            betas = extract(self.betas, t, x_t.shape)
            alphas = 1 - betas
            alphas_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

            posterior_ddim_coef = torch.sqrt(alphas - alphas_cumprod) - sqrt_one_minus_alphas_cumprod
            mean = (x_t + posterior_ddim_coef * noise) / torch.sqrt(alphas)
        else:
            # predict mean using x_0 and noise
            sqrt_alphas_cumprod_prev = extract(self.sqrt_alphas_cumprod_prev, t, x_0.shape)
            sqrt_one_minus_alphas_cumprod_prev = extract(self.sqrt_one_minus_alphas_cumprod_prev, t, x_0.shape)

            mean = sqrt_alphas_cumprod_prev * x_0 + sqrt_one_minus_alphas_cumprod_prev * noise

        return mean

    def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0, return_model_output=False):
        N, C, H, W = x.shape
        assert t.shape == (N,)
        model_output = model(x, t)

        # Learned or fixed variance?
        if self.is_implicit:
            var = log_var = None
        else:
            var, log_var = self.model_var_log_var(self.model_var_type)

            var     = extract(var, t, x.shape) * torch.ones_like(x)
            log_var = extract(log_var, t, x.shape) * torch.ones_like(x)

        # Mean parametrization
        _maybe_clip = lambda x_: x_.clamp(min=-1, max=1) if clip_denoised else x_

        if self.model_mean_type == 'xprev':
            # the model predicts x_{t-1}
            pred_x0 = _maybe_clip(self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output))
            mean    = model_output
        elif self.model_mean_type == 'xstart':
            # the model predicts x_0
            pred_x0_ = model_output
            pred_x0 = _maybe_clip(pred_x0_)
            if not self.is_implicit:
                mean = self.q_posterior_mean(x_0=pred_x0, x_t=x, t=t)
            else:
                # the next step prediction is different than DDPM
                xstart_noise = self.predict_noise_from_start(pred_x0_, x, t)
                mean = self.q_posterior_mean_implicit(x_0=pred_x0, x_t=x, t=t, noise=xstart_noise)
        elif self.model_mean_type == 'eps':
            # the model predicts epsilon
            pred_noise = model_output
            pred_x0_ = self.predict_start_from_noise(x_t=x, t=t, noise=pred_noise)
            pred_x0 = _maybe_clip(pred_x0_)
            if not self.is_implicit:
                mean = self.q_posterior_mean(x_0=pred_x0, x_t=x, t=t)
            else:
                # the next step prediction is different than DDPM
                mean = self.q_posterior_mean_implicit(x_0=pred_x0, x_t=x, t=t, noise=pred_noise)
        elif self.model_mean_type == 'dualx':
            mean, pred_x0 = self.predict_dualx(model_output=model_output, x_t=x, t=t, mode='mean',
                                               is_implicit=self.is_implicit, return_pred_x0=True, fn=_maybe_clip)
        else:
            raise NotImplementedError(f"{self.model_mean_type} is not supported")

        out = (mean, var, log_var,
               pred_x0 if return_pred_x0 else None,
               model_output if return_model_output else None
               )
        return out

    def predict_start_from_noise(self, x_t, t, noise):
        x_0 = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
               - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
        return x_0

    def predict_start_from_prev(self, x_t, t, x_prev):
        x_0 = (extract(1./self.posterior_mean_coef1, t, x_t.shape) * x_prev -
               extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)
        return x_0

    def predict_noise_from_start(self, x_0, x_t, t):
        noise = (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0) \
                / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return noise

    def p_sample(self, model, x, t, noise_fn, *, clip_denoised=True, return_pred_x0=False, return_model_output=False):

        mean, _, log_var, pred_x0, model_output = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            return_pred_x0=return_pred_x0,
            return_model_output=return_model_output,
        )

        if self.is_implicit:
            sample = mean
        else:
            noise = noise_fn(x.shape, dtype=x.dtype, device=x.device)

            shape        = [x.shape[0]] + [1] * (x.ndim - 1)
            nonzero_mask = (t != 0).type(torch.float32).view(*shape).to(x.device)
            sample       = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        out = {'sample': sample}
        if return_pred_x0:
            out['pred_x0'] = pred_x0
        if return_model_output:
            out['model_output'] = model_output
        return out

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, noise_fn=torch.randn,
                      clip_denoised=True, verbose=False):

        img = noise_fn(shape, device=device)

        trange = list(reversed(range(self.num_timesteps)))
        if verbose:
            trange = tqdm(trange, leave=False, desc='p_sample_loop')

        for i in trange:
            out = self.p_sample(
                model=model,
                x=img,
                t=torch.full((shape[0],), i, dtype=torch.int64, device=device),
                noise_fn=noise_fn,
                return_pred_x0=False,
                clip_denoised=clip_denoised,
            )
            img = out['sample'].clone()
            del out
        return img

    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, device, noise_fn=torch.randn, include_x0_pred_timesteps=50,
                                  clip_denoised=True, return_dualx_thr=False, verbose=False):
        assert isinstance(include_x0_pred_timesteps, (int, list))

        if isinstance(include_x0_pred_timesteps, int):
            include_timesteps = list(range(0, self.num_timesteps, include_x0_pred_timesteps))
            num_recorded_x0_pred = self.num_timesteps // include_x0_pred_timesteps
        else:
            include_timesteps = include_x0_pred_timesteps
            num_recorded_x0_pred = len(include_x0_pred_timesteps)

        x0_preds_ = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32)
        thr_preds_ = torch.zeros((shape[0], num_recorded_x0_pred, 1, *shape[2:]), dtype=torch.float32)

        img = noise_fn(shape, dtype=torch.float32, device=device)

        trange = list(reversed(range(self.num_timesteps)))
        if verbose:
            trange = tqdm(trange, leave=False, desc='p_sample_loop_progressive')

        for i in trange:
            # Sample p(x_{t-1} | x_t) as usual
            out = self.p_sample(
                model=model,
                x=img,
                t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                return_pred_x0=True,
                clip_denoised=clip_denoised,
                return_model_output=return_dualx_thr,
            )
            img = out['sample']
            pred_x0 = out['pred_x0']

            # Keep track of prediction of x0
            if i in include_timesteps:
                idx = include_timesteps.index(i)
                x0_preds_[:, idx, ...] = pred_x0.cpu()

                if return_dualx_thr:
                    thr_preds_[:, idx, ...] = out['model_output'][:, 6:7].cpu()

        if return_dualx_thr:
            return img, x0_preds_, thr_preds_
        else:
            return img, x0_preds_

    def predict_dualx(self, model_output, x_t, t, mode=None, is_implicit=False, return_pred_x0=False, fn=None):
        fn = default(fn, lambda x_: x_)

        model_output_xstart, model_output_eps, model_output_thr = model_output[:, :3], model_output[:, 3:6], model_output[:, 6:7]
        pred_x0_xstart = model_output_xstart
        pred_x0_eps = self.predict_start_from_noise(x_t, t, model_output_eps)

        s = model_output_thr.sigmoid()
        pred_x0_thr_ = s * pred_x0_xstart.detach() + (1-s) * (pred_x0_eps).detach()

        pred_x0_thr = fn(pred_x0_thr_)
        if not is_implicit:
            model_output_thr_mean = self.q_posterior_mean(x_0=pred_x0_thr, x_t=x_t, t=t)
        else:
            pred_noise_thr = self.predict_noise_from_start(pred_x0_thr_, x_t, t)
            model_output_thr_mean = self.q_posterior_mean_implicit(x_0=pred_x0_thr, x_t=x_t, t=t, noise=pred_noise_thr)

        if mode == 'mean':
            if not return_pred_x0:
                return model_output_thr_mean
            else:
                return model_output_thr_mean, pred_x0_thr
        else:
            assert mode is None
            pred = torch.cat((model_output_xstart, model_output_eps, model_output_thr_mean), dim=1)
            return pred

    @torch.no_grad()
    def compute_mse_losses(self, x_0, t, noise, x_t, mean, model_output):
        """Compute independent losses for visualization"""
        pred_x_0 = {
            'xprev':    lambda: self.predict_start_from_prev(x_t=x_t, t=t, x_prev=model_output),
            'xstart':   lambda: model_output,
            'eps':      lambda: self.predict_start_from_noise(x_t=x_t, t=t, noise=model_output),
            'dualx':    lambda: model_output[:, :3],
        }[self.model_mean_type]()

        pred_noise = {
            'xprev':    lambda: self.predict_noise_from_start(x_0=pred_x_0, x_t=x_t, t=t),
            'xstart':   lambda: self.predict_noise_from_start(x_0=model_output, x_t=x_t, t=t),
            'eps':      lambda: model_output,
            'dualx':    lambda: model_output[:, 3:6],
        }[self.model_mean_type]()

        pred_mean = {
            'xprev':    lambda: model_output,
            'xstart':   lambda: self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t),
            'eps':      lambda: self.q_posterior_mean(x_0=pred_x_0, x_t=x_t, t=t),
            'dualx':    lambda: self.predict_dualx(model_output=model_output, x_t=x_t, t=t, mode='mean'),
        }[self.model_mean_type]()

        loss_dict = {
            'loss_x_0': torch.mean((x_0 - pred_x_0.detach()).view(x_0.shape[0], -1)**2, dim=1),
            'loss_noise': torch.mean((noise - pred_noise.detach()).view(x_0.shape[0], -1)**2, dim=1),
            'loss_mean': torch.mean((mean - pred_mean.detach()).view(x_0.shape[0], -1) ** 2, dim=1),
        }

        return loss_dict

    def training_losses(self, model, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        # Calculate the loss
        model_output = model(x_t, t)
        mean = self.q_posterior_mean(x_0=x_0, x_t=x_t, t=t)

        target = {
            'xprev':    lambda: mean,
            'xstart':   lambda: x_0,
            'eps':      lambda: noise,
            'dualx':    lambda: torch.cat((x_0, noise, mean), dim=1),
        }[self.model_mean_type]()

        if self.model_mean_type != 'dualx':
            assert model_output.shape[1] == 3
            pred = model_output
        else:
            assert model_output.shape[1] == 7
            pred = self.predict_dualx(model_output=model_output, x_t=x_t, t=t)

        losses = torch.mean((target - pred).view(x_0.shape[0], -1)**2, dim=1)

        with torch.no_grad():
            loss_dict = self.compute_mse_losses(x_0, t, noise, x_t, mean, model_output)

        return losses, loss_dict
