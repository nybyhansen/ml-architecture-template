from src.models import BaseMLModel
from src.metrics import BaseMetric
from src.datatypes import ModellingData
from src.valiadator import BaseValidator
from typing import List

# train models (potentially more than one)
# validate performance
# track list of metrics and plots
# upload model to registry

class ModelBuilder:
    """Component to build, validate"""
    def __init__(self, model: BaseMLModel, metrics: List[BaseMetric], validator: BaseValidator, modelling_data: ModellingData):
        self.model = model
        self.metrics = metrics
        self.validator = validator
        self.modelling_data = modelling_data

    def _validate_model_with_validation_scheme(self):
        
        for validation_fold in self.validator.folds(self.modelling_data):

            self.model.fit(validation_fold.training_data)

            predictions = self.model.predict(validation_fold.validation_data.features_data)
            ground_truths = validation_fold.validation_data.targets_data

    def _compute_metrics():
        pass

    def _train_full_model(self) -> None:
        self.model.fit(self.modelling_data)
    
    def _upload_to_registry():
        pass

    def build(self):
        self._validate_model_with_validation_scheme()
        self._train_full_model()
        self._upload_to_registry()
