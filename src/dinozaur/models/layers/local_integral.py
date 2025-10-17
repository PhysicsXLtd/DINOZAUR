"""LocalIntegral with Linear MLP kernel definition."""

import torch
import torch_scatter.segment_csr as scatter_segment_csr


class LocalIntegral(torch.nn.Module):
    """LocalIntegral with Linear MLP kernel."""

    def __init__(self, in_channels, out_channels, mlp):
        """Initialize LocalIntegral.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mlp: Initialized MLP class for kernel computation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = mlp

    def forward(
        self,
        x: torch.Tensor,
        in_points: torch.Tensor,
        out_points: torch.Tensor,
        neighbors_index: torch.Tensor,
        neighbors_row_splits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape: [b, n, c], where b is the batch size, n is the number of input
                points and c is the number of channels.
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
            Output features, shape: [b, m, c_out], where c_out is determined by the message
            function.
        """
        if neighbors_index.shape[0] != 1 or neighbors_row_splits.shape[0] != 1:
            raise NotImplementedError(
                "Message passing supports batching only when the neighbors are the same "
                "across the batch."
            )

        x = x.view(x.shape[0], -1, x.shape[-1])
        in_points = in_points.view(in_points.shape[0], -1, in_points.shape[-1])

        neighbors_index = neighbors_index.to(torch.long)
        neighbors_row_splits = neighbors_row_splits.to(torch.long)
        num_neighbors = torch.diff(neighbors_row_splits)

        x = x[:, neighbors_index[0]]
        in_points = in_points[:, neighbors_index[0]]
        out_points = torch.repeat_interleave(out_points, num_neighbors[0], dim=1)

        in_points, out_points = torch.broadcast_tensors(in_points, out_points)
        in_out = torch.cat([in_points, out_points], dim=-1)
        kernel = self.mlp(in_out)

        if kernel.dim() == x.dim() - 1:
            return kernel.unsqueeze(-1) * x  # Scalar kernel
        messages = kernel * x  # Vector kernel

        return scatter_segment_csr(messages, neighbors_row_splits, reduce="mean")
