"""Initialize the training module."""

from dinozaur.training.scalers import GaussianScaler, IdentityScaler

ScalerType = IdentityScaler | GaussianScaler
