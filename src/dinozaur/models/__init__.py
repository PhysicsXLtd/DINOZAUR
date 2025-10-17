"""Initialize the models module."""

from dinozaur.models.bayesian_dinozaur_d import BayesianDINOZAURd
from dinozaur.models.dinozaur import DINOZAUR
from dinozaur.models.dinozaur_d import DINOZAURd
from dinozaur.models.fno import FNO
from dinozaur.models.gino_d import GINOd

ModelType = FNO | GINOd | DINOZAUR | DINOZAURd | BayesianDINOZAURd

__all__ = [
    "BayesianDINOZAURd",
    "DINOZAUR",
    "DINOZAURd",
    "FNO",
    "GINOd",
]
