"""Modelling package for ML Assignment 2.

Split into three modules so each concern can be tested on its own:
    datasets.py     loading and constraint validation
    classifiers.py  preprocessing pipeline and the five estimators
    evaluation.py   cross-validated metrics
"""

from . import classifiers, datasets, evaluation

__all__ = ["datasets", "classifiers", "evaluation"]
