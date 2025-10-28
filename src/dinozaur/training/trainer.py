"""Trainer class definition."""

import importlib
import logging
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinozaur.models import ModelType
from dinozaur.training import ScalerType
from dinozaur.training.dataset import H5Dataset
from dinozaur.training.losses import ELBOLoss
from dinozaur.training.metrics import RunningMetrics, get_metrics

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)


def load_class(class_path: str):
    """Helper function to load class by its name.

    Args:
        class_path: Full path of the class to import.

    Returns:
        Unintialized class.
    """
    class_name = class_path.split(".")[-1]
    class_path = ".".join(class_path.split(".")[:-1])
    return getattr(importlib.import_module(class_path), class_name)


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        training_params: dict,
        model_params: dict,
        data_params: dict,
    ):
        """Initialize Trainer.

        Args:
            training_params: Training parameters.
            model_params: Model architecture parameters.
            data_params: Dataset parameters.
        """
        # initialize training parameters
        self.training_params: dict = training_params
        self.device: str = training_params["device"]
        self.num_epochs: int = training_params["num_epochs"]
        self.supports_sampling: bool = training_params.get("supports_sampling", False)

        # initialize acrhitecture
        self.model_params: dict = model_params

        model_class = load_class(self.model_params["architecture"])
        self.model: ModelType = model_class(**self.model_params["architecture_params"]).to(
            self.device
        )

        # initialize dataset
        self.data_params: dict = data_params

        self.train_loader = DataLoader(
            dataset=H5Dataset(
                f"{data_params['data_folder']}/train",
                features=data_params["features"],
                target=data_params["target"],
                extra_inputs=data_params["extra_inputs"],
                device=self.device,
            ),
            batch_size=data_params["batch_size"],
            shuffle=True,
        )

        self.val_loader = DataLoader(
            dataset=H5Dataset(
                f"{data_params['data_folder']}/test",
                features=data_params["features"],
                target=data_params["target"],
                extra_inputs=data_params["extra_inputs"],
                device=self.device,
            ),
            batch_size=data_params["batch_size"],
            shuffle=False,
        )
        self.x_normalizer: ScalerType = load_class(data_params["x_normalizer_class"])(
            self.train_loader, "x"
        )
        self.y_normalizer: ScalerType = load_class(data_params["y_normalizer_class"])(
            self.train_loader, "target"
        )

        # initialize training classes
        optimizer_class = load_class(self.training_params["optimizer"])
        self.optimizer_has_train_mode = (
            self.training_params["optimizer"].split(".")[-1] == "AdamWScheduleFree"
        )
        self.optimizer: torch.optim.Optimizer = optimizer_class(
            self.model.parameters(), **self.training_params["optimizer_params"]
        )

        if self.training_params["scheduler"] is not None:
            self.scheduler_per_batch = (
                self.training_params["scheduler"].split(".")[-1] == "CyclicLR"
            ) or (self.training_params["scheduler"].split(".")[-1] == "OneCycleLR")
            scheduler_class = load_class(self.training_params["scheduler"])
            if self.training_params["scheduler"].split(".")[-1] == "OneCycleLR":
                self.training_params["scheduler_params"]["total_steps"] = self.num_epochs * len(
                    self.train_loader
                )
            self.scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler_class(
                self.optimizer, **self.training_params["scheduler_params"]
            )
        else:
            self.scheduler_per_batch = False
            self.scheduler = None

        criterion_class = load_class(self.training_params["criterion"])
        self.criterion: torch.nn.modules.loss._Loss = criterion_class(
            **self.training_params["criterion_params"]
        )

    def normalize_batch_data(self, batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply data normalization."""
        batch_data["x"] = self.x_normalizer.transform(batch_data["x"])
        batch_data["target"] = self.y_normalizer.transform(batch_data["target"])
        return batch_data

    def train(self):
        """Full training loop."""
        logger.info(f"Training of {self.model.__class__.__name__} started.")
        logger.info(
            "Number of trainable parameters: "
            + f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        for epoch in range(self.num_epochs):
            self.train_running_loss = 0
            self.train_one_epoch()

            self.val_running_loss = 0
            self.overall_total_metrics = RunningMetrics()
            self.eval_one_epoch()
            if (not self.scheduler_per_batch) and (self.scheduler is not None):
                self.scheduler.step()
            val_metrics = self.overall_total_metrics.aggregate()

            logger.info(
                f"Epoch {epoch}, train loss={self.train_running_loss / len(self.train_loader)}, "
                + f"val loss={self.val_running_loss / len(self.val_loader)}."
            )
        if self.supports_sampling:
            self.overall_total_metrics = RunningMetrics()
            self.eval_uq(n_samples=100)
            val_metrics = self.overall_total_metrics.aggregate()
        logger.info("Training finished.")
        metrics_str = ", ".join(f"{key}={value.item()}" for key, value in val_metrics.items())
        logger.info(f"Test metrics are: {metrics_str}.")

    def train_one_epoch(self):
        """One training dataloader loop function."""
        self.model.train()
        if self.optimizer_has_train_mode:
            self.optimizer.train()
        for batch, batch_data in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            batch_data = self.normalize_batch_data(batch_data)

            labels = batch_data.pop("target")
            prediction = self.model(**batch_data)

            if isinstance(self.criterion, ELBOLoss):
                loss = self.criterion(
                    prediction.squeeze(0).view(1, -1),
                    labels.squeeze(0).view(1, -1),
                    self.model.variance.exp(),
                    self.model.entropy,
                )
            else:
                loss = self.criterion(prediction, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler_per_batch:
                self.scheduler.step()

            self.train_running_loss += loss.item()

    def eval_one_epoch(self):
        """One evaluation dataloader loop function."""
        self.model.eval()
        if self.optimizer_has_train_mode:
            self.optimizer.eval()
        with torch.no_grad():
            for batch, batch_data in enumerate(tqdm(self.val_loader)):
                metrics_this_step = {}

                batch_data = self.normalize_batch_data(batch_data)

                labels = batch_data.pop("target")

                if isinstance(self.criterion, ELBOLoss):
                    predictions = self._generate_samples(batch_data, n_samples=5)
                    prediction = predictions.mean(0)
                    prediction_var = predictions.var(0)
                    val_loss = self.criterion(prediction, labels, prediction_var)
                else:
                    prediction = self.model(**batch_data)
                    val_loss = self.criterion(prediction, labels)

                labels = self.y_normalizer.reverse_transform(labels)
                prediction = self.y_normalizer.reverse_transform(prediction)

                metrics = get_metrics(
                    prediction.detach(),
                    labels.detach(),
                    None,
                )
                for key, value in metrics.items():
                    metrics_this_step[f"val.{key}"] = value
                self.overall_total_metrics.append(metrics_this_step)

                self.val_running_loss += val_loss.item()

    def eval_uq(self, n_samples: int = 100):
        """One evaluation dataloader loop function."""
        with torch.no_grad():
            self.model.eval()
            self._manage_dropout(
                enable=self.model_params["architecture_params"].get("dropout", False) > 0
            )

            for batch, batch_data in enumerate(tqdm(self.val_loader)):
                metrics_this_step = {}

                batch_data = self.normalize_batch_data(batch_data)

                labels = batch_data.pop("target")
                predictions = self._generate_samples(batch_data, n_samples=n_samples)

                labels = self.y_normalizer.reverse_transform(labels)
                predictions = self.y_normalizer.reverse_transform(predictions)

                prediction = predictions.mean(0)
                prediction_std = predictions.std(0)

                metrics = get_metrics(
                    prediction.detach(),
                    labels.detach(),
                    prediction_std.detach(),
                )
                for key, value in metrics.items():
                    metrics_this_step[f"val.{key}"] = value
                self.overall_total_metrics.append(metrics_this_step)

    def _generate_samples(
        self, batch_data: dict[str, torch.Tensor], n_samples: int = 5
    ) -> torch.Tensor:
        """Helper function to draw samples from predictive posterior distribution.

        Args:
            batch_data: Data dictionary containing inpits to the model.
            n_samples: Number of samples to draw. Defaults to 5.

        Returns:
            Tensor containing drawn samples, shape: [n_samples, spatial_dims, output_size].
        """
        log_var = torch.tensor(-4)
        posterior_predictions = []

        for _ in range(n_samples):
            posterior_sample = self.model(**batch_data)
            epsilon = torch.randn_like(posterior_sample)
            if hasattr(self.model, "variance"):
                log_var = self.model.variance
            posterior_sample = posterior_sample + torch.sqrt(log_var.exp()) * epsilon
            posterior_predictions.append(posterior_sample)

        samples_tensor = torch.cat(posterior_predictions, dim=0)
        return samples_tensor

    def predict(self, data_sample: dict[str, torch.Tensor]):
        """Helper function to perform inference."""
        with torch.no_grad():
            self.model.eval()
            self._manage_dropout(
                enable=self.model_params["architecture_params"].get("dropout", False) > 0
            )
            sample = {}

            data_sample.pop(self.val_loader.dataset.target, None)
            features = torch.cat([data_sample[key] for key in self.val_loader.dataset.features], -1)

            sample["x"] = features
            for key, value in self.val_loader.dataset.extra_inputs.items():
                sample[key] = data_sample[value]
            for key, value in sample.items():
                sample[key] = value.to(self.device)

            sample["x"] = self.x_normalizer.transform(sample["x"])
            prediction = self.model(**sample)
            prediction = self.y_normalizer.reverse_transform(prediction)
        return prediction

    def _manage_dropout(self, enable: bool) -> None:
        """Function to enable dropout layers at inference.

        Based on function from https://github.com/ENSTA-U2IS/torch-uncertainty/.

        Args:
            enable: Whether to enable (True) or disable (False) dropout layers
        """
        # filter all modules whose class name starts with `Dropout`
        filtered_modules = set()
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                filtered_modules.add(m)

        for m in filtered_modules:
            if enable:
                m.train()
            else:
                m.eval()
