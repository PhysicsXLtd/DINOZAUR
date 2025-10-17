"""Initialize the layers module."""

from dinozaur.models.layers.multipliers import (
    BayesianDiffusionMultiplier,
    DiffusionMultiplier,
    TensorMultiplier,
)

MultiplierType = TensorMultiplier | DiffusionMultiplier | BayesianDiffusionMultiplier
