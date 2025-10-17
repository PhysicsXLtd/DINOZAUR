"""Bayesian DINOZAUR decoder model definition."""

from typing import Literal

import torch

from dinozaur.models.layers.blocks import DoubleSkipBlock, NoSkipBlock
from dinozaur.models.layers.fourier_convolution import FourierConvolution
from dinozaur.models.layers.local_integral import LocalIntegral
from dinozaur.models.layers.mlp import MLP
from dinozaur.models.layers.multipliers import BayesianDiffusionMultiplier
from dinozaur.models.layers.padding import Padding, Unpadding
from dinozaur.models.layers.positional_encoding import PositionalEncoding


class BayesianDINOZAURd(torch.nn.Module):
    """Bayesian DINOZAUR decoder architecture."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int = 32,
        n_blocks: int = 5,
        *,
        modes: list[int],
        extent: list[float] = [2 * torch.pi],
        include_gradient_features: bool = False,
        full_rank: bool = True,
        prior_mean: float = 0.01,
        prior_scale: float = 1.0,
        init_mean: float = -5.0,
        init_scale: float = 0.5,
        n_positional_encoding_frequencies: int = 0,
        domain_padding: float | list[float] = 0.0,
        padding_mode: Literal["symmetric", "one_sided"] = "one_sided",
        dropout: float = 0.0,
        activation: str = "GELU",
    ):
        """Initialize BayesianDINOZAURd.

        Args:
            input_size: Number of channels in the input field.
            output_size: Number of channels in the output field.
            modes: Number of Fourier modes in each dimension that are transformed in the FNO blocks.
                The length should be equal to number of dimensions, and each element less or equal
                to the corresponding grid size.
            width: Number of channels in the hidden layers throughout the network. Defaults to 32.
            n_blocks: Number of blocks. Defaults to 4.
            extent: A list of domain extents. Used for rescaling the frequencies.
                Defaults to [2 * torch.pi].
            include_gradient_features: Whether to include the spatial gradient features.
                Defaults to False.
            full_rank: Whether to use full rank covariance matrix. Defaults to True.
            prior_mean: Prior distribution mean. Defaults to 0.01.
            prior_scale: Prior distribution scale. Defaults to 1.0.
            init_mean: Initial posterior mean. Defaults to -5.0.
            init_scale: Initial posterior scale. Defaults to 0.5.
            n_positional_encoding_frequencies: Number of positional encoding frequencies.
                If 0, no positional encoding is used. Defaults to 0.
            domain_padding: Fraction of the domain size that is padded with zeros.
                If float (default), padding applied to all sides equally. If list[float],
                padding for each dimension should be specified. Defaults to 0.0.
            padding_mode: Whether to pad the domain symmetrically or only on one side in each
                dimension. Defaults to "one_sided".
            dropout: Dropout rate applied to MLPs in DoubleSkip blocks. Defaults to 0.0.
            activation: Activation function used in the MLPs throughout the network.
                Must be a name of a PyTorch activation function in torch.nn. Defaults to "GELU".
        """
        super().__init__()

        space_dim = len(modes)

        self.variance = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.constant_(self.variance, -3.0)

        self.positional_encoding = PositionalEncoding(
            num_frequencies=n_positional_encoding_frequencies,
            include_self=True,
        )
        self.nu_positional_encoding = PositionalEncoding(
            num_frequencies=n_positional_encoding_frequencies,
            include_self=True,
        )
        self.u_positional_encoding = PositionalEncoding(
            num_frequencies=n_positional_encoding_frequencies,
            include_self=True,
        )

        self.lifting = MLP(
            in_channels=input_size + input_size * 2 * n_positional_encoding_frequencies,
            out_channels=width,
            hidden_channels=[width * 2],
            activation=activation,
        )

        self.pad = Padding(
            domain_padding=domain_padding,
            padding_mode=padding_mode,
        )

        self.double_skip_blocks = torch.nn.ModuleList(
            [
                DoubleSkipBlock(
                    activation=activation,
                    fourier_convolution=FourierConvolution(
                        in_channels=width,
                        out_channels=2 * width if include_gradient_features else width,
                        modes=modes,
                        extent=extent,
                        include_gradient_features=include_gradient_features,
                        multiplier=BayesianDiffusionMultiplier(
                            in_channels=width,
                            out_channels=width,
                            modes=modes,
                            full_rank=full_rank,
                            prior_mean=prior_mean,
                            prior_scale=prior_scale,
                            init_mean=init_mean,
                            init_scale=init_scale,
                        ),
                    ),
                    mlp=MLP(
                        in_channels=2 * width if include_gradient_features else width,
                        out_channels=width,
                        hidden_channels=[max(1, round(width * 0.5))],
                        dropout=dropout,
                        activation=activation,
                    ),
                )
                for _ in range(n_blocks - 1)
            ]
        )

        self.unpad = Unpadding(
            domain_padding=domain_padding,
            padding_mode=padding_mode,
        )

        self.no_skip_block = NoSkipBlock(
            local_integral=LocalIntegral(
                in_channels=width,
                out_channels=width,
                mlp=MLP(
                    in_channels=2 * space_dim * (1 + 2 * n_positional_encoding_frequencies),
                    out_channels=width,
                    hidden_channels=[width * 2],
                    activation=activation,
                ),
            ),
            mlp=MLP(
                in_channels=width,
                out_channels=width,
                hidden_channels=[width * 2],
                activation=activation,
            ),
        )

        self.projection = MLP(
            in_channels=width,
            out_channels=output_size,
            hidden_channels=[width * 2] * 2,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_points: torch.Tensor,
        out_points: torch.Tensor,
        u_nu_neighbor_index: torch.Tensor,
        u_nu_neighbor_row_splits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Note:
            Batching is supported only for the same point cloud.

        Args:
            x: Input field, shape:
                [n_batch, *grid_shape, input_size].
            out_points: Output points, shape:
                [n_batch, out_n_vertices, space_dim].
            grid_points: Grid points, shape:
                [n_batch, *grid_shape, space_dim].
            u_nu_neighbor_index: Neighbor index for uniform to non-uniform transformation, shape:
                [1, e], where e is the total number of edges
            u_nu_neighbor_row_splits: Row splits for uniform to non-uniform transformation, shape:
                [1, out_n_vertices + 1], where m is the number of output points.

        Returns:
            Output tensor of shape:
                [n_batch, out_n_vertices, output_size].
        """
        x = self.positional_encoding(x)

        nu_encoded_points = self.nu_positional_encoding(out_points)

        u_encoded_points = self.u_positional_encoding(grid_points)

        x = self.lifting(x)
        x = self.pad(x=x)

        for block in self.double_skip_blocks:
            x = block(x)

        x = self.unpad(x=x)

        x = self.no_skip_block(
            x,
            in_points=u_encoded_points,
            out_points=nu_encoded_points,
            neighbors_index=u_nu_neighbor_index,
            neighbors_row_splits=u_nu_neighbor_row_splits,
        )

        return self.projection(x)

    @property
    def entropy(self) -> torch.Tensor:
        """Gather KL regularisation terms from layers."""
        total_entropy = torch.tensor(0.0, device=self.variance.device)

        filtered_modules = set()
        for m in self.modules():
            if isinstance(m, BayesianDiffusionMultiplier):
                filtered_modules.add(m)

        for m in filtered_modules:
            total_entropy += m.entropy
        return total_entropy
