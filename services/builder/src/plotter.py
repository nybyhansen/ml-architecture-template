from abc import ABC, abstractmethod
from numpy import ndarray
from src.models import BaseMLModel
import plotly
from typing import Optional


class BasePlotter(ABC):
    """Interface for plots"""

    @abstractmethod
    def create_plot(
        self,
        y_pred: ndarray,
        y_true: ndarray,
        model: Optional[BaseMLModel] = None,
    ) -> plotly.graph_objects.Figure:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass