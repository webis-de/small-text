from small_text.stopping_criteria.base import DeltaFScore
from small_text.stopping_criteria.change import ClassificationChange
from small_text.stopping_criteria.kappa import KappaAverage


__all__ = [
    'ClassificationChange',
    'DeltaFScore',
    'KappaAverage'
]
