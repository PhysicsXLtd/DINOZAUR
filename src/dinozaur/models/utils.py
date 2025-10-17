"""Util functions definitions."""

from typing import cast

import open3d as o3d
import torch


def get_activation(activation: str) -> type[torch.nn.Module]:
    """Retrieve an activation function from `torch.nn` by name.

    Args:
        activation: Name of the activation function as a string.

    Raises:
        ValueError: If the specified activation function is not found in `torch.nn`.

    Returns:
        A callable activation function from `torch.nn`.
    """
    activation_fn = getattr(torch.nn, activation, None)
    if activation_fn is None:
        raise ValueError(f"Activation function '{activation}' is not available in torch.nn")
    return cast(type[torch.nn.Module], activation_fn)


def fixed_radius_search(
    data: torch.Tensor, queries: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds neighbors within the radius.

    The inputs can have a batch dimension of 1, in which case the outputs will have it too.

    Args:
        data: Set of possible neighbors, shape [n, d]
        queries: Point for which to find neighbors, shape [m, d]
        radius: Radius of each neighborhood.

    Returns:
        A tuple of:
            neighbors_index: torch.Tensor with dtype=torch.int64
                Index of each neighbor in data for every point
                in queries. Neighbors are ordered in the same orderings
                as the points in queries. Implementations can differ by a permutation of the
                neighbors for every point.
            neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                The value at index j is the sum of the number of
                neighbors up to query point j-1. First element is 0
                and last element is the total number of neighbors.
    """
    # Note: Open3d's fixed-radius search works with any dimension, not just 3d.
    has_batch_dim = data.ndim == 3
    if has_batch_dim:
        data = data[0]
        queries = queries[0]

    o3d_data = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    o3d_queries = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(queries))

    search = o3d.core.nns.NearestNeighborSearch(
        dataset_points=o3d_data,
        index_dtype=o3d.core.Dtype.Int64,
    )
    search.fixed_radius_index(radius=radius)
    o3d_neighbors_index, _, o3d_neighbors_row_splits = search.fixed_radius_search(
        query_points=o3d_queries, radius=radius
    )
    neighbors_index = torch.utils.dlpack.from_dlpack(o3d_neighbors_index.to_dlpack())
    neighbors_row_splits = torch.utils.dlpack.from_dlpack(o3d_neighbors_row_splits.to_dlpack())

    if has_batch_dim:
        neighbors_index = neighbors_index.unsqueeze(0)
        neighbors_row_splits = neighbors_row_splits.unsqueeze(0)

    return neighbors_index, neighbors_row_splits
