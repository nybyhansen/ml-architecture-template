# ML ARCHITECTURE TEMPLATE

## Desired features
- Made for batch jobs
- Serverless architecture (AWS) (SAM)
- SOLID and made for change
- MLops best practices

## Functionality
- Training machine learning models
- Deploying models to an end-point (serving)
- Model registry with models stored (mlflow)
- Monitor training performance
- Monitor live performance
- Detect drifts for input features
- Model Evaluation (compare models, behavioral tests)


## Components

### 1. ModelBuilder
consume:
- metrics
- models
- data

produce:
- artifacts
- logs

direct:


### 2. ModelEvaluator



___

# Overview of ML project structure
We have three parts of this ML solution all devided into one repository each:
1. Data Pipeline (repo)
2. Machine Learning (repo)
3. Presenter Client (repo) (optional)