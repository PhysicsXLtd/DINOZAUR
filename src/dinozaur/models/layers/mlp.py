"""MLP layer definition."""

import torch

from dinozaur.models.utils import get_activation


class MLP(torch.nn.Sequential):
    """MLP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int],
        bias: bool = True,
        end_with_activation: bool = False,
        dropout: float = 0.0,
        activation: str = "GELU",
    ):
        """Initialize MLP.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Widths of hidden channels.
            bias: Whether to include bias in linear layers. Defaults to True.
            end_with_activation: Whether to apply activation and dropout in the last layer.
                Defaults to False.
            dropout: Probability for dropout layers. Defaults to 0.
            activation: Activation function which will be stacked on top of the linear layer.
                Defaults to "GELU".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Construct the channel configuration for all layers
        channels = [in_channels] + list(hidden_channels) + [out_channels]
        num_layers = len(channels) - 1

        layers: list[torch.nn.Module] = []

        # Build each layer
        for i in range(num_layers):
            layer: list[torch.nn.Module] = []

            # Linear layer
            linear_layer = torch.nn.Linear(channels[i], channels[i + 1], bias=bias)

            # Append the linear layer
            layer.append(linear_layer)

            # Add activation and dropout if not the last layer or specified
            if i != num_layers - 1 or end_with_activation:
                layer.append(get_activation(activation)())
                if dropout > 0.0:
                    layer.append(torch.nn.Dropout(p=dropout))

            # Append the constructed layer
            layers += layer

        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input field, shape:
                n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels]
                or [n_batch, in_n_vertices, in_channels].

        Returns:
            Result of the forward function, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, out_channels]
                or [n_batch, in_n_vertices, out_channels].

        """
        return super().forward(x)
