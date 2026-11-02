# Churn Model Release Checklist

## Data and training
- Record dataset version, target definition, feature list, and split strategy.
- Confirm preprocessing is fitted only on training data.
- Check for target leakage and duplicate customers across splits.
- Store reproducible random seeds and dependency versions where applicable.

## Evaluation
Record baseline and candidate metrics on an untouched evaluation set. Include confusion matrix, precision, recall, F1, ROC-AUC where suitable, and threshold rationale.

## API and application
Validate request schemas, missing fields, invalid values, and predictable error responses. Do not expose training data, secrets, or stack traces in API responses.

## Operations
Document model artifact location, configuration, health checks, logs, and rollback steps. Monitor input drift and prediction distributions; schedule performance review when ground truth becomes available.

## Release gate
Do not describe the model as production-ready until tests, security review, deployment, monitoring, and rollback have been verified.
