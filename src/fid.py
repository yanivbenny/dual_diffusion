from typing import Any, Callable, List, Optional, Union

import torch
import torchmetrics as tm
from torch import Tensor
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.image.fid import dim_zero_cat, sqrtm
from torchmetrics.utilities import rank_zero_info, rank_zero_warn

from src.torch_utils import to_device


def _compute_mean_cov(features):
    features = dim_zero_cat(features)
    # computation is extremely sensitive so it needs to happen in double precision
    features = features.double()

    # calculate mean and covariance
    n = features.shape[0]
    mean = features.mean(dim=0)
    diff = features - mean
    cov = 1.0 / (n - 1) * diff.t().mm(diff)

    return mean, cov


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6):
    r"""
    Adjusted version of https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(
            f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    tr_sigma1 = torch.trace(sigma1)
    tr_sigma2 = torch.trace(sigma2)

    mean_component = diff.dot(diff)
    var_component = tr_sigma1 + tr_sigma2 - 2 * tr_covmean
    var_component1 = tr_sigma1 - tr_sigma2
    var_component2 = 2 * (tr_sigma2 - tr_covmean)

    return {'FID': mean_component + var_component,
            'FID_mean_component': mean_component,
            'FID_var_component1': var_component1,
            'FID_var_component2': var_component2
            }


class EnhancedFID(tm.FID):
    def __init__(
        self,
        store_device=None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.store_device = store_device

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        features = self.inception(imgs)

        if self.store_device is not None:
            features = to_device(features, self.store_device)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        real_mean, real_cov = _compute_mean_cov(self.real_features)
        fake_mean, fake_cov = _compute_mean_cov(self.fake_features)

        # compute fid
        fid_dict = _compute_fid(real_mean, real_cov, fake_mean, fake_cov)
        return to_device(fid_dict, self.device)
