"""Metrics utils definitions."""

from collections import defaultdict

import torch
from uncertainty_toolbox.metrics import interval_score, miscalibration_area, nll_gaussian

from dinozaur.training.losses import RelativeLpLoss


def get_metrics(
    y_pred: torch.Tensor, y: torch.Tensor, y_pred_std: torch.Tensor | None = None, norm_dims="auto"
) -> dict[str, torch.Tensor]:
    """Helper function to compute metrics.

    Args:
        y_pred: Predicted tensor.
        y: Target tensor.
        y_pred_std: Standard deviation of a predicted tensor. Defaults to None.
        norm_dims: Specifies which dimension or dimensions of input to calculate the norm
                across.

    Returns:
        Dictionary containing metrics for a given sample.
    """
    rl2 = RelativeLpLoss(norm_dims=[1, 2], reduction="none")
    metrics_one_sample = {}
    metrics_one_sample["MSE"] = torch.mean((y_pred - y) ** 2, dim=list(torch.arange(1, y.ndim)))
    rl2_value = rl2(y_pred, y).squeeze()
    if rl2_value.ndim > 2:
        rl2_value = rl2_value.mean(-1)
    metrics_one_sample["RL2"] = rl2_value
    if y_pred_std is not None:
        y = y.detach().cpu().numpy().squeeze()
        y_pred = y_pred.detach().cpu().numpy().squeeze()
        y_pred_std = y_pred_std.detach().cpu().numpy().squeeze()

        metrics_one_sample["IS"] = torch.tensor(interval_score(y_pred, y_pred_std, y))
        metrics_one_sample["McalA"] = torch.tensor(miscalibration_area(y_pred, y_pred_std, y))
        metrics_one_sample["NLL"] = torch.tensor(nll_gaussian(y_pred, y_pred_std, y))
    return metrics_one_sample


class RunningMetrics:
    """Class for tracking metrics during training."""

    def __init__(self):
        """Initialize RunningMetrics."""
        self.metrics = defaultdict(list)
        self.statistics = {
            "mean": torch.nanmean,
        }

    def append(self, metrics_this_step: dict):
        """Append metrics from the current step to the global dict."""
        for name, value in metrics_this_step.items():
            self.metrics[name].append(torch.atleast_1d(value))

    def aggregate(self) -> dict:
        """Create aggregated metrics."""
        stacked_metrics = {name: torch.cat(values) for name, values in self.metrics.items()}
        return {
            f"{name}.{statistic}": func(value)
            for name, value in stacked_metrics.items()
            for statistic, func in self.statistics.items()
        }
