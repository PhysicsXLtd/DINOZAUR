"""FourierConvolution layer definition."""

from typing import cast

import torch

from dinozaur.models.layers import MultiplierType
from dinozaur.models.layers.mlp import MLP


class FourierConvolution(torch.nn.Module):
    r"""Uniform to uniform spectral convolution.

    An instance of Fourier convolution $g(y) =  \mathcal F^{-1} \lbrack \mathcal{F}
    \lbrack K\rbrack \odot  \mathcal F [f] \rbrack (y)$, where both $\mathcal F$ and
    $\mathcal F^{-1}$ map from/to a uniformly-spaced set of points.
    """

    wavenumbers: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        multiplier: MultiplierType,
        modes: list[int],
        extent: list[float] = [2 * torch.pi],
        include_gradient_features: bool = False,
    ):
        """Initialize FourierConvolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            multiplier: Initialized multiplier class applied in the spectral space.
            modes: Number of lowest frequency Fourier modes in each dimension.
                The last number of modes will be halved automatically.
            extent: A list of domain extents. Used for rescaling the frequencies.
                Defaults to [2 * torch.pi].
            include_gradient_features: Whether to include the spatial gradient features.
                Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multiplier = multiplier
        self.modes = modes
        self.n_dims = len(modes)
        self.weight_modes = [self.modes[i] for i in range(self.n_dims - 1)] + [
            self.modes[-1] // 2 + 1
        ]
        if len(extent) == self.n_dims:
            self.extent = extent
        else:
            self.extent = extent * self.n_dims
        self.include_gradient_features = include_gradient_features
        self.multiplier_out_channels = multiplier.out_channels

        if include_gradient_features:
            self.spatial_gradient_feature_weights = MLP(
                in_channels=self.multiplier_out_channels,
                out_channels=self.multiplier_out_channels,
                hidden_channels=[],
                bias=False,
            )

        wavenumbers_list = [torch.fft.fftfreq(size, 1 / size) for size in self.weight_modes[:-1]]

        wavenumbers_list.append(
            torch.fft.rfftfreq(
                (self.weight_modes[-1] - 1) * 2, 1 / ((self.weight_modes[-1] - 1) * 2)
            )
        )
        wavenumbers = (
            2
            * torch.pi
            * torch.cartesian_prod(*wavenumbers_list).T.reshape(1, self.n_dims, *self.weight_modes)
            / torch.Tensor(self.extent).reshape(1, self.n_dims, *[1] * self.n_dims)
        )
        self.register_buffer(
            "wavenumbers",
            wavenumbers,
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x: input field, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, in_channels].

        Returns:
             Result of the forward function, shape:
                [n_batch, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, out_channels].
        """
        input_spatial_shape = x.shape[1:-1]
        x = x.permute(0, -1, *range(1, x.dim() - 1))

        spectrum = torch.fft.rfftn(
            x,
            dim=list(range(-self.n_dims, 0)),
            norm="forward",
        )

        slices = self._compute_spectrum_slices(spectrum)
        spectrum = self._convolve(spectrum, slices)

        if self.include_gradient_features:
            spectrum = self._concat_gradients_to_spectrum(spectrum, slices)

        y = torch.fft.irfftn(
            spectrum, dim=list(range(-self.n_dims, 0)), s=input_spatial_shape, norm="forward"
        )

        if self.include_gradient_features:
            y = self._compute_gradient_features(y)
        return y.permute(0, *range(2, y.dim()), 1)

    def _convolve(self, spectrum: torch.Tensor, slices: list[slice]) -> torch.Tensor:
        """Convolution operation in basis space.

        Args:
            spectrum: Transformed spectrum of input field, shape:
                [n_batch, out_channels,
                    spatial_dim_1, spatial_dim_2, ..., spatial_dim_d // 2 + 1].
            slices: Slices of the spectrum to apply the multiplier to.

        Returns:
            Dictionary containing result of convolution in basis space.
        """
        convolved_spectrum = torch.zeros(
            (spectrum.shape[0], self.multiplier_out_channels, *spectrum.shape[2:]),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        if self.n_dims > 1:
            spectrum = torch.fft.fftshift(spectrum, dim=list(range(-self.n_dims, -1)))

        x = spectrum[slices]

        # multipliers expect channels last, hence permuting the x and convolved_spectrum
        # here, we applied torch.fft.fftshift to spectrum, so we need to do the same to eigenvalues
        convolved_spectrum[slices] = self.multiplier(
            x=x.permute(0, *range(2, x.dim()), 1),
            eigenvalues=torch.fft.fftshift(
                (self.wavenumbers**2).sum(1), dim=list(range(-self.n_dims, -1))
            ),
        ).permute(0, -1, *range(1, x.dim() - 1))

        if self.n_dims > 1:
            convolved_spectrum = torch.fft.ifftshift(
                convolved_spectrum, dim=list(range(-self.n_dims, -1))
            )
        return convolved_spectrum

    def _compute_spectrum_slices(self, spectrum: torch.Tensor) -> list[slice]:
        """Slices to reduce the spectrum tensor before multiplication of weight.

        Args:
            spectrum: Spectrum of input field, shape:
                [n_batch, in_channels, to_basis_modes_1, to_basis_modes_2, ..., to_basis_modes_d].

        Returns:
            Slices to reduce the spectrum tensor.
        """
        spectrum_modes = list(spectrum.shape[2:])
        starts = [
            (size - min(size, n_mode)) for (size, n_mode) in zip(spectrum_modes, self.weight_modes)
        ]
        slices = [slice(None), slice(None)]
        slices += [
            slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]
        ]
        slices += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        return slices

    def _compute_spectral_gradients(
        self, spectrum: torch.Tensor, wavenumbers: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral gradients.

        Args:
            spectrum: Spectrum of input field, shape:
                [n_batch, d, channels, modes_1, modes_2, ..., modes_d].
            wavenumbers: Wavenumbers, shape:
                [n_batch, d, 1, modes_1, modes_2, ..., modes_d]

        Returns:
            Gradients of the spectrum, shape:
                [n_batch, d, channels, modes_1, modes_2, ..., modes_d]
        """
        return spectrum * 1.0j * wavenumbers

    def _concat_gradients_to_spectrum(
        self, spectrum: torch.Tensor, slices: list[slice]
    ) -> torch.Tensor:
        """Compute gradients in spectral domain and concatenate them to spectrum.

        Args:
            spectrum: Transformed spectrum of input field, shape:
                [n_batch, out_channels,
                    spatial_dim_1, spatial_dim_2, ..., spatial_dim_d // 2 + 1].
            slices: Slices of the spectrum to compute the gradients for.

        Returns:
            Spectrum and gradients in spectral domain, shape:
                [n_batch, (d + 1) * out_channels,
                    spatial_dim_1, spatial_dim_2, ..., spatial_dim_d // 2 + 1].
        """
        n_batches, n_channels = spectrum.shape[0], spectrum.shape[1]

        gradients = torch.zeros(
            (
                spectrum.shape[0],
                self.n_dims * self.multiplier_out_channels,
                *spectrum.shape[2:],
            ),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        if self.n_dims > 1:
            spectrum = torch.fft.fftshift(spectrum, dim=list(range(-self.n_dims, -1)))

        x = spectrum[slices].unsqueeze(
            1
        )  # n_batches, 1, channels, spatial_dim_1, ..., spatial_dim_d // 2 + 1

        _wavenumbers = cast(torch.Tensor, self.wavenumbers)
        wavenumbers = torch.fft.fftshift(
            _wavenumbers.unsqueeze(2), dim=list(range(-self.n_dims, -1))
        )  # n_batches, d, 1, spatial_dim_1, ..., spatial_dim_d // 2 + 1

        gradients[slices] = self._compute_spectral_gradients(x, wavenumbers).reshape(
            n_batches, self.n_dims * n_channels, *self.weight_modes
        )  # n_batches, d * channels, spatial_dim_1, ..., spatial_dim_d // 2 + 1

        if self.n_dims > 1:
            spectrum = torch.fft.ifftshift(spectrum, dim=list(range(-self.n_dims, -1)))
            gradients = torch.fft.ifftshift(gradients, dim=list(range(-self.n_dims, -1)))

        spectrum_and_gradients = torch.cat(
            (spectrum, gradients), dim=1
        )  # n_batches, (d + 1) * channels, spatial_dim_1, ..., spatial_dim_d // 2 + 1

        return spectrum_and_gradients

    def _compute_gradient_features(self, field_and_gradients: torch.Tensor) -> torch.Tensor:
        """Apply learnable weights to gradients to compute the gradient features.

        Args:
            field_and_gradients: Field and gradients in physical domain, shape:
                [n_batch, (d + 1) * out_channels, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d]
                or [n_batch, (d + 1) * out_channels, n_vertices].

        Returns:
            Field and gradient features, shape:
                [n_batch, 2 * out_channels, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d]
                or [n_batch, 2 * out_channels, n_vertices].
        """
        n_batch = field_and_gradients.shape[0]
        spatial_dims = field_and_gradients.shape[2:]

        field, gradients = (
            field_and_gradients[:, : self.multiplier_out_channels],
            field_and_gradients[:, self.multiplier_out_channels :]
            .reshape(n_batch, self.n_dims, self.multiplier_out_channels, *spatial_dims)
            .permute(0, 1, *range(3, len(spatial_dims) + 3), 2),
        )  # gradients: n_batch, d, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d, n_channels

        x_gradient_features = (
            (self.spatial_gradient_feature_weights(gradients) * gradients)
            .sum(1)
            .permute(0, -1, *range(1, len(spatial_dims) + 1))
        )  # n_batch, n_channels, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d

        field = torch.cat(
            (field, torch.tanh(x_gradient_features)), dim=1
        )  # n_batch, 2 * n_channels, spatial_dim_1, spatial_dim_2, ..., spatial_dim_d

        return field
