from abc import ABC, abstractmethod
from src.datatypes import ModellingData, InferenceData

class BaseMLModel(ABC):

    @abstractmethod
    def fit(modelling_data: ModellingData) -> None:
        pass

    @abstractmethod
    def predict(inference_data: InferenceData):
        pass
