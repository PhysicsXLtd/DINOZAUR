"""Training script."""

#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import torch
import yaml

from dinozaur.training.trainer import Trainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.INFO)

experiment = sys.argv[1] if len(sys.argv) > 1 else ""

if experiment == "":
    logger.info("Usage: python train.py path/to/config.yml")
    sys.exit(1)

with open(experiment) as f:
    config = yaml.safe_load(f)

logger.info(f"Starting experiment {experiment}.")

trainer = Trainer(**config)

trainer.train()

experiment_name = Path(experiment).with_suffix("")
logger.info(f"Saving model to logs/{experiment_name}.")
Path(f"logs/{experiment_name}").mkdir(parents=True, exist_ok=True)
torch.save(trainer, f"logs/{experiment_name}/model.pt")
