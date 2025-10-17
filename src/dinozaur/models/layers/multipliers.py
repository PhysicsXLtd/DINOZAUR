"""Multiplier layer definitions."""

import logging
from typing import Literal

import numpy as np
import torch
import torch.distributions as dist
from tltorch import CPTensor, TTTensor, TuckerTensor
from tltorch.factorized_tensors.core import FactorizedTensor
from torch.distributions.transforms import (
    LowerCholeskyTransform,
    SoftplusTransform,
)

_TUCKER_EINSUM_EQUATIONS = [
    "kio,Kk,Ii,Oo,BKI->BKO",
    "klio,Kk,Ll,Ii,Oo,BKLI->BKLO",
    "klmio,Kk,Ll,Mm,Ii,Oo,BKLMI->BKLMO",
]

_TT_EINSUM_EQUATIONS = [
    "aKb,bIc,cOf,BKI->BKO",
    "aKb,bLc,cId,dOe,BKLI->BKLO",
    "aKb,bLc,cMd,dIe,eOf,BKLMI->BKLMO",
]

_CP_EINSUM_EQUATIONS = [
    "H,KH,IH,OH,BKI->BKO",
    "H,KH,LH,IH,OH,BKLI->BKLO",
    "H,KH,LH,MH,IH,OH,BKLMI->BKLMO",
]


class TensorMultiplier(torch.nn.Module):
    """TensorMultiplier."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: list[int],
        halve_last_mode: bool = True,
        factorization: Literal["dense", "tucker", "tt", "cp"] = "dense",
        factorization_rank: float | None = None,
    ):
        """Initialize TensorMultiplier.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of lowest frequency Fourier modes in each dimension.
            halve_last_mode: Whether to the number of modes for the last dimension is halved
                (due to the data being real-valued). Defaults to True.
            factorization: Factorization type for spectral convolution weight, defaults to "dense"
                meaning no factorization is applied.. Defaults to "dense".
            factorization_rank: The fraction of parameters the factorization preserves, if applied.
                Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        n_dims = len(self.modes)

        if halve_last_mode:
            weight_modes = [self.modes[i] for i in range(n_dims - 1)] + [self.modes[-1] // 2 + 1]
        else:
            weight_modes = self.modes

        init_sd = (2 / (self.in_channels + self.out_channels)) ** 0.5

        weight_shape = (*weight_modes, self.in_channels, self.out_channels)

        if factorization == "dense":
            self.weights = torch.nn.Parameter(
                torch.randn(
                    weight_shape,
                    dtype=torch.cfloat if halve_last_mode else torch.float,
                )
                * init_sd
            )
        else:
            self.weights = FactorizedTensor.new(
                weight_shape,
                rank=factorization_rank,
                factorization=factorization,
                dtype=torch.cfloat if halve_last_mode else torch.float,
            )
            self.weights.normal_(0, init_sd)

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor | None = None) -> torch.Tensor:
        """Forward function.

        Args:
            x: Reduced spectrum of input field: shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, in_channels].
            eigenvalues: Defaults to None.

        Returns:
            Contraction, shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, out_channels].
        """
        # return multiply_factorized_weights(self.weights, x)
        n_dims = len(x.shape) - 2
        if n_dims <= 0:  # pragma: no cover
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected a batch dimension, 1 to 3 mode "
                f" dimensions and a channel dimension."
            )

        if n_dims > 3:
            logging.warning(
                "Optimized multiplication of factorized tensors not implemented for > 3 dimensions."
                "Using a naive einsum."
            )
            return torch.einsum("...I,...IO->...O", x, self.weights)

        if isinstance(self.weights, TuckerTensor):
            return torch.einsum(
                _TUCKER_EINSUM_EQUATIONS[n_dims - 1], self.weights.core, *self.weights.factors, x
            )
        elif isinstance(self.weights, TTTensor):
            return torch.einsum(_TT_EINSUM_EQUATIONS[n_dims - 1], *self.weights.factors, x)
        elif isinstance(self.weights, CPTensor):
            return torch.einsum(
                _CP_EINSUM_EQUATIONS[n_dims - 1], self.weights.weights, *self.weights.factors, x
            )
        return torch.einsum("...I,...IO->...O", x, self.weights)


class DiffusionMultiplier(torch.nn.Module):
    """DiffusionMultiplier."""

    def __init__(self, in_channels: int, out_channels: int, modes: list[int]):
        """Initialize DiffusionMultiplier.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of lowest frequency Fourier modes in each dimension.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dims = len(self.modes)
        self.diffusion_time = torch.nn.Parameter(torch.empty(self.in_channels))
        torch.nn.init.zeros_(self.diffusion_time)

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: Basis coefficients of input field: shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, in_channels].
            eigenvalues: Eigenvalues, shape:
                [n_batch, n_eigenvalues] or
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d].

        Returns:
            Diffused basis coefficients, shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, in_channels].
        """
        with torch.no_grad():
            self.diffusion_time.data.clamp_(min=1e-8)

        time = self.diffusion_time.reshape(*[1] * self.n_dims, -1)
        diffusion_coefs = torch.exp(-eigenvalues.unsqueeze(-1) * time)
        x_diffuse_spectral = diffusion_coefs * x

        return x_diffuse_spectral


class BayesianDiffusionMultiplier(torch.nn.Module):
    """BayesianDiffusionMultiplier."""

    entropy: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: list[int],
        full_rank: bool = True,
        prior_mean: float = 0.01,
        prior_scale: float = 1.0,
        init_mean: float = -5.0,
        init_scale: float = 0.5,
    ):
        """Initialize BayesianDiffusionMultiplier.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of lowest frequency Fourier modes in each dimension.
            full_rank: Whether to use full rank covariance matrix. Defaults to True.
            prior_mean: Prior distribution mean. Defaults to 0.01.
            prior_scale: Prior distribution scale. Defaults to 1.0.
            init_mean: Initial posterior mean. Defaults to -5.0.
            init_scale: Initial posterior scale. Defaults to 0.5.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dims = len(self.modes)
        self.full_rank = full_rank

        # Mean of VI posterior
        self.init_mean = init_mean
        self.diffusion_time_posterior_mean = torch.nn.Parameter(torch.empty(self.in_channels))
        torch.nn.init.constant_(self.diffusion_time_posterior_mean, self.init_mean)

        # Scale of VI posterior covariance
        self.init_scale = init_scale
        self.diffusion_time_posterior_scale = torch.nn.Parameter(torch.empty(self.in_channels))
        torch.nn.init.constant_(self.diffusion_time_posterior_scale, self.init_scale)

        # VI posterior covariance parameterisation, through lower Cholesky decomposition
        if full_rank:
            self.diffusion_time_posterior_scale_tril = torch.nn.Parameter(
                torch.eye(self.in_channels)
            )
            torch.nn.init.constant_(self.diffusion_time_posterior_scale_tril, init_scale)
            self.transform_to_lowercholesky = LowerCholeskyTransform()

        else:
            self.transform_scale = SoftplusTransform()

        # Define prior
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.prior = dist.LogNormal(np.log(self.prior_mean), self.prior_scale)

        # The variational approx. and the prior needs to have the
        # same support (positive reals), implemented using torch's bijector
        self.transform = dist.constraint_registry.biject_to(self.prior.support)

        # Register the KL regularisation term
        self.register_buffer("entropy", torch.empty(1), persistent=False)

    def _variational_posterior(self) -> dist.Distribution:
        """Variational approximate posterior from which we sample time.

        Returns:
            torch `Normal` or `MultivariateNormal` distribution object.
        """
        mean = self.diffusion_time_posterior_mean
        scale = self.diffusion_time_posterior_scale
        base_distribution: dist.Normal | dist.MultivariateNormal
        if self.full_rank:
            scale_tril = self.transform_to_lowercholesky(
                scale[..., None] * self.diffusion_time_posterior_scale_tril
            )
            base_distribution = dist.MultivariateNormal(mean, scale_tril=scale_tril)
        else:
            scale_positive = self.transform_scale(scale)
            base_distribution = dist.Normal(mean, scale=scale_positive)

        # The variational approx. and the prior needs to have
        # the same support (positive reals)
        return dist.TransformedDistribution(
            base_distribution=base_distribution, transforms=self.transform
        )

    def _prior(self, positive_time: torch.Tensor) -> torch.Tensor:
        """Place a log normal prior on `t`.

        Args:
            positive_time: The diffusion times vector, shape: [width] or [n_batch, width]

        Returns:
            Log prior density summed over all channels, shape: [] or [n_batch]
        """
        log_density = self.prior.log_prob(positive_time).sum(dim=-1)
        return log_density

    def _entropy(self, positive_time: torch.Tensor, posterior: dist.Distribution) -> None:
        """Calculate the KL regularisation term.

        Args:
            positive_time: The diffusion times vector, shape: [width] or [n_batch, width]
            posterior: The posterior distribution created by self._variational_posterior()

        Returns:
            None, but sets self.entropy to the Kullback-Liebler divergence between the prior
                and the variational approx, shape [] or [n_batch]
        """
        prior = self._prior(positive_time)
        self.entropy = posterior.log_prob(positive_time).sum(dim=-1) - prior

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Forward function .

        Args:
            x: Basis coefficients of input field: shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, in_channels].
            eigenvalues: Eigenvalues, shape:
                [n_batch, n_eigenvalues] or
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d].

        Returns:
            Diffused basis coefficients, shape:
                [n_batch, weight_modes_1, weight_modes_2, ..., weight_modes_d, in_channels].
        """
        posterior = self._variational_posterior()
        diffusion_time = posterior.rsample()
        with torch.no_grad():
            diffusion_time.data.clamp_(min=1e-8)

        time = diffusion_time.reshape(*[1] * self.n_dims, -1)
        diffusion_coefs = torch.exp(-eigenvalues.unsqueeze(-1) * time)
        x_diffuse_spectral = diffusion_coefs * x

        self._entropy(diffusion_time, posterior)

        return x_diffuse_spectral
