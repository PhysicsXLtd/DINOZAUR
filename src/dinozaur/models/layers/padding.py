"""Padding layer definition."""

from typing import Literal

import torch


class Padding(torch.nn.Module):
    """Padding."""

    def __init__(
        self,
        domain_padding: float | list[float],
        padding_mode: Literal["symmetric", "one_sided"] = "one_sided",
    ):
        """Initialize Padding.

        Args:
            domain_padding: Percentage of padding to use, greater than zero, smaller than 1.
                Must be a list in case of multiple spatial dimensions.
            padding_mode: Whether to pad on both sides. Defaults to "one_sided".
        """
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies domain padding scaled automatically to the input resolution.

        Args:
            x: Input field, shape:
                [n_batches, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels].

        Returns:
            Padded field, shape:
                [n_batches,
                spatial_dim_1 + padding_1,
                spatial_dim_2 + padding_2,
                ...,
                spatial_dim_d + padding_d,
                in_channels].
        """
        x = x.permute(0, -1, *range(1, x.dim() - 1))

        resolution = x.shape[2:]

        if isinstance(self.domain_padding, float):
            domain_padding = [self.domain_padding] * len(resolution)
        elif isinstance(self.domain_padding, list):
            domain_padding = self.domain_padding
        else:
            raise NotImplementedError("domain_padding has to be either list of float!")

        if len(domain_padding) != len(resolution):
            raise ValueError(
                "domain_padding length must match the number of spatial/time dimensions "
                "(excluding batch, channel dimensions)"
            )

        padding = [round(p * r) for (p, r) in zip(domain_padding, resolution)][::-1]

        if self.padding_mode == "symmetric":
            padding = [i for p in padding for i in (p, p)]
        else:
            padding = [i for p in padding for i in (0, p)]

        x = torch.nn.functional.pad(x, padding, mode="constant")

        x = x.permute(0, *range(2, x.dim()), 1).contiguous()

        return x


class Unpadding(torch.nn.Module):
    """Unpadding."""

    def __init__(
        self,
        domain_padding: float | list[float],
        padding_mode: Literal["symmetric", "one_sided"] = "one_sided",
    ):
        """Initialize Unpadding.

        Args:
            domain_padding: Percentage of padding to use, greater than zero, smaller than 1.
                Must be a list in case of multiple spatial dimensions.
            padding_mode: Whether to pad on both sides. Defaults to "one_sided".
        """
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the padding from padded inputs.

        Args:
            x: Padded field, shape:
                [n_batches,
                spatial_dim_1 + padding_1,
                spatial_dim_2 + padding_2,
                ...,
                spatial_dim_d + padding_d,
                in_channels].

        Returns:
            Unpadded field, shape:
                [n_batches, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels].
        """
        x = x.permute(0, -1, *range(1, x.dim() - 1))

        resolution = x.shape[2:]

        if isinstance(self.domain_padding, float):
            domain_padding = [self.domain_padding] * len(resolution)
        elif isinstance(self.domain_padding, list):
            domain_padding = self.domain_padding
        else:
            raise NotImplementedError("domain_padding has to be either list of float!")

        if len(domain_padding) != len(resolution):
            raise ValueError(
                "domain_padding length must match the number of spatial/time dimensions "
                "(excluding batch, channel dimensions)"
            )

        padding_multiplier = 2 if self.padding_mode == "symmetric" else 1
        padding = [
            round(p * r)
            for (p, r) in zip(
                domain_padding,
                [
                    round(r / (1 + padding_multiplier * p))
                    for (p, r) in zip(domain_padding, resolution)
                ],
            )
        ]

        unpad_list = []
        for p in padding:
            if p == 0:
                padding_end = None
                padding_start = None
            else:
                padding_end = p if self.padding_mode == "symmetric" else None
                padding_start = -p
            unpad_list.append(slice(padding_end, padding_start, None))
        unpad_indices = (Ellipsis,) + tuple(unpad_list)

        x = x[unpad_indices]

        x = x.permute(0, *range(2, x.dim()), 1).contiguous()

        return x
