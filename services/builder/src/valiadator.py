import numpy as np
from abc import ABC, abstractmethod
from src.datatypes import ValidationFold, ModellingData

class BaseValidator(ABC):
    @abstractmethod
    def folds():
        pass


class TrainTestSplitValidator(BaseValidator):
    """Validator to make a simple split into a training and validation set"""
    def folds(input_data: ModellingData):
        """Yields sets of training and validation folds"""

        TRAININGSET_SIZE_PERCENTAGE = 0.2
        N_ROWS = input_data.training_data.shape[0]

        training_indices = np.random.rand(N_ROWS) < TRAININGSET_SIZE_PERCENTAGE

        # training modelling data
        training_modelling_data = ModellingData(
            features_data = input_data.features_data.iloc[training_indices],
            target_data = input_data.target_data.iloc[training_indices]
        )

        # validation modelling data
        validation_modelling_data = ModellingData(
            features_data = input_data.features_data.iloc[~training_indices],
            target_data = input_data.target_data.iloc[~training_indices]
        )

        yield ValidationFold(
            training_data=training_modelling_data,
            validation_data=validation_modelling_data,
        )
