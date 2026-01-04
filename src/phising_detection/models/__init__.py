"""
Phishing detection models package.

This package provides a modular framework for training and evaluating
phishing detection models.

Modules:
    - data_prep: Data loading and preparation
    - model_configs: Model definitions and training
    - evaluation: Model evaluation and metrics
    - visualization: Plotting and visualizations
"""

from .model_utils import data_prep
from .model_utils import model_configs
from .model_utils import evaluation
from .model_utils import visualization

__all__ = ['data_prep', 'model_configs', 'evaluation', 'visualization']
