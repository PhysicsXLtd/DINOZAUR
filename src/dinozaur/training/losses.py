"""Loss class definitions."""

from typing import Literal

import torch


class RelativeLpLoss(torch.nn.modules.loss._Loss):
    """Relative LpLoss class."""

    def __init__(
        self,
        norm_dims: int | tuple[int, ...] | list[int] | Literal["auto"] = "auto",
        norm_order: float = 2,
        reduction: Literal["sum", "mean", "none"] = "mean",
    ):
        """Initialize RelativeLpLoss.

        Args:
            norm_dims: Specifies which dimension or dimensions of input to calculate the norm
                across.
            norm_order: Order of the norm. Defaults to 2.
            reduction: Type of reduction to apply. Defaults to 'mean'.
        """
        super().__init__(reduction=reduction)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Reduction {reduction} not supported.")
        self.norm_dims = norm_dims
        self.norm_order = norm_order

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss.

        Args:
            input: Predictions tensor.
            target: Target tensor.

        Returns:
            Loss value.
        """
        norm_dims: int | tuple[int, ...] | list[int] | None
        if self.norm_dims == "auto":
            norm_dims = tuple(range(1, input.dim()))
        else:
            norm_dims = self.norm_dims

        diff_norm = torch.norm(
            input - target,
            p=self.norm_order,
            dim=norm_dims,
            keepdim=False,
        )
        target_norm = torch.norm(
            target,
            p=self.norm_order,
            dim=norm_dims,
            keepdim=False,
        )

        if torch.any(target_norm == 0):
            raise RuntimeError("RelativeLpLoss is not compatible with target zero norm.")
        loss = diff_norm / target_norm
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ELBOLoss(torch.nn.GaussianNLLLoss):
    """Evidence lower bound loss."""

    def __init__(self, full: bool = True, reduction: str = "sum"):
        """Initialize ELBOLoss.

        Args:
            full: Whether to  include the constant term in the loss calculation. Defaults to True.
            reduction: Type of reduction to apply. Defaults to 'sum'.
        """
        super().__init__(full=full, reduction=reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        var: torch.Tensor | float,
        entropy: torch.Tensor | float = 0,
    ) -> torch.Tensor:
        """Forward pass of the loss.

        Args:
            input: Predictions tensor.
            target: Target tensor.
            var: Predictions variance.
            entropy: Kullback-Liebler divergence between the prior and the variational
                approximation.

        Returns:
            Loss value.
        """
        nll = super().forward(input=input, target=target, var=var)
        return nll + entropy
