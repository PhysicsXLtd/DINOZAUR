"""PositionalEncoding layer definition."""

import torch


class PositionalEncoding(torch.nn.Module):
    """PositionalEncoding.

    The implementation follows the paper
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    """

    frequencies: torch.Tensor

    def __init__(
        self, num_frequencies: int, include_self: bool = True, base_frequency: float | None = 1e-4
    ):
        """Initialize PositionalEncoding.

        Args:
            num_frequencies: Number of frequencies to use for positional encoding.
            include_self: Whether to include the input in the output. Defaults to True.
            base_frequency: Base frequency of sinusoidal functions. Defaults to 1e-4.
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_self = include_self
        self.base_frequency = base_frequency

        exponents = torch.arange(0, self.num_frequencies, dtype=torch.float32)
        # Note: neuraloperator incorrectly divides by 2 here, resulting in min frequency being
        # ~ 1/sqrt(max_positions) = sqrt(base_frequency). We also include -1 to have base_frequency
        # as min frequency exactly.
        exponents /= self.num_frequencies - 1
        frequencies = self.base_frequency**exponents
        frequencies = frequencies.view(1, 1, -1).to(dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: Input points, shape: [..., in_channels] where ... can be any number of dimensions.
               Supports both batched and non-batched input automatically.

        Returns:
            Result of the forward function: shape: [..., out_channels].
                If `include_self=False`, the output will only contain the positional encoding of
                    input channels.
                If `include_self=True`, the output will contain the input channels that were
                    encoded and their positional encoding.
        """
        x_skip = x

        x = x.unsqueeze(-1) * self.frequencies
        x = torch.cat([x.sin(), x.cos()], dim=-2)
        x = x.flatten(-2)

        if self.include_self:
            x = torch.cat([x_skip, x], dim=-1)

        return x
