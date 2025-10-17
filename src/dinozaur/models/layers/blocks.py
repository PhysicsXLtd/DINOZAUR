"""Block layer definitions."""

import torch

from dinozaur.models.layers.fourier_convolution import FourierConvolution
from dinozaur.models.layers.local_integral import LocalIntegral
from dinozaur.models.layers.mlp import MLP
from dinozaur.models.utils import get_activation


class DoubleSkipBlock(torch.nn.Module):
    r"""DoubleSkipBlock.

    Performs $\operatorname{f_{ob}}(x, x') =
    \operatorname{MLP}( \operatorname{activation}(W_1(x) + x')) + W_2(x)$,
    where $W_1$ and $W_2$ are linear layers.
    """

    def __init__(self, fourier_convolution: FourierConvolution, mlp: MLP, activation: str = "GELU"):
        """Initialize DoubleSkipBlock.

        Args:
            fourier_convolution: Initialized Fourier convolution class.
            mlp: Initialized MLP class.
            activation: Activation function to use. Must be a class name from torch.nn.
                Defaults to "GELU".
        """
        super().__init__()
        self.mlp = mlp
        self.fourier_convolution = fourier_convolution
        self.activation = get_activation(activation)()

        self.norm_layers = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(fourier_convolution.out_channels),
                torch.nn.LayerNorm(mlp.out_channels),
            ]
        )
        self.w1 = MLP(
            in_channels=fourier_convolution.in_channels,
            out_channels=fourier_convolution.out_channels,
            hidden_channels=[],
            bias=False,
        )
        self.w2 = MLP(
            in_channels=fourier_convolution.in_channels,
            out_channels=self.mlp.out_channels,
            hidden_channels=[],
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input field, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels]
                or [n_batch, in_n_vertices, in_channels].

        Returns:
            Result of the forward function, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, out_channels]
                or [n_batch, out_n_vertices, out_channels].

        """
        x_transformed = self.fourier_convolution(x)
        x_transformed = self.norm_layers[0](x_transformed)
        x_skip_1 = self.w1(x)
        x_skip_2 = self.w2(x)
        x = x_transformed + x_skip_1
        x = self.activation(x)
        x = self.mlp(x)
        x = x + x_skip_2
        x = self.norm_layers[1](x)
        return self.activation(x)


class NoSkipBlock(torch.nn.Module):
    r"""NoSkipBlock.

    Performs  $\operatorname{f_{ob}}(x, x') = \operatorname{MLP}(x')$.
    """

    def __init__(self, local_integral: LocalIntegral, mlp: MLP, activation: str = "Identity"):
        """Initialize NoSkipBlock.

        Args:
            local_integral: Initialized local integral transform class.
            mlp: Initialized MLP class.
            activation: Activation function to use. Must be a class name from torch.nn.
                Defaults to "GELU".
        """
        super().__init__()
        self.mlp = mlp
        self.local_integral = local_integral
        self.activation = get_activation(activation)()

        self.norm_layers = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(local_integral.out_channels),
                torch.nn.LayerNorm(mlp.out_channels),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        in_points: torch.Tensor,
        out_points: torch.Tensor,
        neighbors_index: torch.Tensor,
        neighbors_row_splits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input field, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels]
                or [n_batch, in_n_vertices, in_channels].
            in_points: Input points with shape: [b, e, d] or [b, 1, n, d], where e is
                the total number of edges.
            out_points: Output points with shape: [b, e, d] or [b, m, 1, d], where e is
                the total number of edges.
            neighbors_index: Index of neighbors for each query point, shape: [1, e], where e is
                the total number of edges. Note that we only support batching with the same point
                cloud.
            neighbors_row_splits: Row splits for neighbors_index, shape: [1, m + 1], where m is the
                number of output points.

        Returns:
            Result of the forward function, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, out_channels]
                or [n_batch, out_n_vertices, out_channels].

        """
        x_transformed = self.local_integral(
            x, in_points, out_points, neighbors_index, neighbors_row_splits
        )
        x_transformed = self.norm_layers[0](x_transformed)
        x = self.mlp(x_transformed)
        x = self.norm_layers[1](x)
        return self.activation(x)
