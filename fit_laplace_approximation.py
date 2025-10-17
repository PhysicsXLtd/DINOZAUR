"""Laplce approximation fitting and evaluation script."""

#!/usr/bin/env python3
import logging
import sys

import torch
from curvature.curvatures import BlockDiagonal as LaplaceApproximation
from tqdm import tqdm

from dinozaur.training.metrics import RunningMetrics, get_metrics
from dinozaur.training.trainer import Trainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)

model_path = sys.argv[1] if len(sys.argv) > 1 else ""

if model_path == "":
    logger.info("Usage: python fit_laplace_approximation.py path/to/checkpoint.pt")
    sys.exit(1)


log_var = -4
add = 1e6
multiply = 5e2
n_samples = 100

trainer: Trainer
trainer = torch.load(model_path, weights_only=False)


logger.info(f"Fitting LA for model {model_path}.")

laplace_model = LaplaceApproximation(trainer.model.projection[-1])

for batch, batch_data in enumerate(tqdm(trainer.train_loader)):
    trainer.model.zero_grad()

    batch_data = trainer.normalize_batch_data(batch_data)

    labels = batch_data.pop("target")
    prediction = trainer.model(**batch_data)

    loss = torch.nn.functional.gaussian_nll_loss(
        prediction,
        labels,
        var=torch.ones_like(labels, device=trainer.device) * torch.tensor(log_var).exp(),
    )

    loss.backward()
    laplace_model.update(batch_size=1)

overall_total_metrics = RunningMetrics()

laplace_model.invert(add=add, multiply=multiply)
trainer.model.eval()

with torch.inference_mode():
    for batch, batch_data in enumerate(tqdm(trainer.val_loader)):
        metrics_this_step = {}

        batch_data = trainer.normalize_batch_data(batch_data)

        labels = batch_data.pop("target")

        posterior_predictions = []
        for _ in range(n_samples):
            laplace_model.sample_and_replace()
            posterior_sample = trainer.model(**batch_data)
            epsilon = torch.randn_like(posterior_sample)
            posterior_sample = posterior_sample + torch.sqrt(torch.tensor(log_var).exp()) * epsilon
            posterior_predictions.append(posterior_sample)
        predictions = torch.cat(posterior_predictions, dim=0)

        labels = trainer.y_normalizer.reverse_transform(labels)
        predictions = trainer.y_normalizer.reverse_transform(predictions)

        prediction = predictions.mean(0)
        prediction_std = predictions.std(0)

        metrics = get_metrics(
            prediction.detach(),
            labels.detach(),
            prediction_std.detach(),
        )
        for key, value in metrics.items():
            metrics_this_step[f"val.{key}"] = value
        overall_total_metrics.append(metrics_this_step)

    val_metrics = overall_total_metrics.aggregate()

    logger.info("Training finished.")
    metrics_str = ", ".join(f"{key}={value.item()}" for key, value in val_metrics.items())
    logger.info(f"Test metrics are: {metrics_str}.")
