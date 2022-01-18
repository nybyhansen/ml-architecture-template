from dataclasses import dataclass
import dataclasses


@dataclass
class ModellingData:
    pass

@dataclass
class InferenceData:
    pass

@dataclass
class ValidationFold:
    training_data: ModellingData
    validation_data: ModellingData
