"""DINOZAUR model definition."""

from typing import Literal

import torch

from dinozaur.models.layers.blocks import DoubleSkipBlock
from dinozaur.models.layers.fourier_convolution import FourierConvolution
from dinozaur.models.layers.mlp import MLP
from dinozaur.models.layers.multipliers import DiffusionMultiplier
from dinozaur.models.layers.padding import Padding, Unpadding
from dinozaur.models.layers.positional_encoding import PositionalEncoding


class DINOZAUR(torch.nn.Module):
    """DINOZAUR architecture."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int = 32,
        n_blocks: int = 4,
        *,
        modes: list[int],
        extent: list[float] = [2 * torch.pi],
        include_gradient_features: bool = False,
        n_positional_encoding_frequencies: int = 0,
        domain_padding: float | list[float] = 0.0,
        padding_mode: Literal["symmetric", "one_sided"] = "one_sided",
        dropout: float = 0.0,
        activation: str = "GELU",
    ):
        """Initialize DINOZAUR.

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
        self.positional_encoding = PositionalEncoding(
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
                        multiplier=DiffusionMultiplier(
                            in_channels=width,
                            out_channels=width,
                            modes=modes,
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
                for _ in range(n_blocks)
            ]
        )

        self.unpad = Unpadding(
            domain_padding=domain_padding,
            padding_mode=padding_mode,
        )
        self.projection = MLP(
            in_channels=width,
            out_channels=output_size,
            hidden_channels=[width * 2],
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: Input field, shape:
                [n_batch, *grid_shape, input_size].

        Returns:
            Result of the forward function, shape:
                [n_batch, *grid_shape, output_size].
        """
        x = self.positional_encoding(x)
        x = self.lifting(x)
        x = self.pad(x=x)
        for block in self.double_skip_blocks:
            x = block(x)
        x = self.unpad(x=x)
        return self.projection(x)
