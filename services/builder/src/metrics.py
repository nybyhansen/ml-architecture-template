from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Union


class BaseMetric(ABC):
    """Base class for performance metrics"""

    @abstractmethod
    def compute(
        self,
        y_true: ndarray,
        y_pred: ndarray,
    ) -> Union[int, float]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns the name of the metric"""
        pass
