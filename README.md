# Patient Readmission Risk Modeling

An end-to-end machine-learning project for predicting whether a diabetic hospital encounter is associated with **readmission within 30 days**.

The project started as a notebook-based modeling study and has been rebuilt into a reproducible training + API pipeline. The rebuilt version keeps the notebook's useful feature engineering and model comparison, while fixing patient leakage, SMOTE leakage, test-set tuning, and training/serving skew.

## Target

The binary target is:

- `1`: `readmitted == "<30"`
- `0`: `readmitted == ">30"` or `"NO"`

The model therefore predicts **readmission within 30 days**, not any future readmission.

## What changed from the original notebook

The original notebook contained strong exploratory work, including:

- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- SMOTE
- 10-fold cross-validation
- medication and hospital-utilization features
- diagnosis grouping
- admission/discharge/source grouping
- log transformations

However, several parts of the experimental design could leak information:

1. `patient_nbr` was removed before splitting, so encounters from the same patient could appear in both train and test.
2. SMOTE was applied before the train/test split and before cross-validation.
3. XGBoost repeatedly used the test set for early stopping and parameter tuning.
4. ROC-AUC was sometimes calculated from hard `0/1` predictions instead of probability scores.
5. Global outlier removal and target-related feature selection were performed before evaluation.

The production pipeline keeps the useful modeling ideas but moves every data-dependent operation into the correct training boundary.

## Corrected methodology

```
Raw encounters
      |
      v
Patient-level stratified split
      |
      +-----------------------------+
      |                             |
      v                             v
Training patients              Untouched test patients
      |
      v
10-fold StratifiedGroupKFold
      |
      +--> fold training
      |       |
      |       +--> feature engineering
      |       +--> one-hot encoding
      |       +--> SMOTE
      |       +--> model
      |
      +--> fold validation
              |
              +--> never SMOTE'd
```

After cross-validation selects the model and hyperparameters:

```
training patients
      +
validation patients
      |
      v
final model
      |
      v
untouched test patients
      |
      v
final metrics
```

### Patient-level splitting

The source data contains multiple encounters for many patients. `patient_nbr` is therefore used only as a grouping variable.

It is never used as a predictive feature.

The split guarantees:

```
train patients ∩ validation patients = empty
train patients ∩ test patients       = empty
validation patients ∩ test patients  = empty
```

### Cross-validation

Model selection uses **10-fold StratifiedGroupKFold**.

This provides two protections:

- `Group`: encounters from the same patient cannot cross folds.
- `Stratified`: the algorithm attempts to preserve the positive/negative target distribution across folds.

ROC-AUC is the model-selection metric because it evaluates probability ranking independently of a particular classification threshold.

### SMOTE

SMOTE is implemented **inside the imbalanced-learn pipeline**:

```
Feature engineering
        ↓
DictVectorizer
        ↓
SMOTE
        ↓
Model
```

Therefore SMOTE is fitted separately inside each CV training fold.

The validation fold is never used to create synthetic observations.

The final test set is also never used by SMOTE.

> Note: standard SMOTE is being applied after one-hot encoding. This is convenient and reproducible, but it can create fractional values in one-hot dimensions. A future experiment can compare this against SMOTENC or another categorical-aware approach.

## Notebook-derived feature engineering

The shared transformer in `src/preprocessing.py` converts the notebook's exploratory transformations into deterministic code.

### Hospital utilization

```text
patient_service =
    number_outpatient
  + number_emergency
  + number_inpatient
```

### Medication features

The pipeline creates:

- `num_med`: number of medications used
- `med_change`: number of medications changed up/down

### Diagnosis grouping

The notebook grouped ICD-9-style diagnosis codes into eight broad clinical categories plus an unknown/other group.

The production transformer performs the same deterministic grouping for:

- primary diagnosis
- secondary diagnosis
- additional diagnosis

### Admission/discharge/source grouping

The notebook reduced related categorical IDs into broader categories. The production transformer applies the same mappings.

### Age

Age ranges are mapped to their midpoint:

```text
[60-70) → 65
[70-80) → 75
...
```

### Log features

The skewed utilization/count variables used in the notebook are transformed with:

```text
log1p(x) = log(1 + x)
```

This reduces the influence of highly right-skewed count distributions.

## Models

The corrected training pipeline compares the model families used in the notebook:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

Each model receives the same feature-engineering and SMOTE framework.

The model with the best grouped-CV ROC-AUC is selected.

No model is selected by repeatedly looking at the final test set.

## Threshold selection

The model outputs:

```
P(readmitted within 30 days | patient data)
```

A probability of 0.5 is not automatically the appropriate decision threshold for an imbalanced classification problem.

After model selection, the separate validation set is used to choose a threshold that maximizes F1.

The test set is not used for threshold selection.

The selected threshold is stored with the serialized model.

## Evaluation

The final untouched test set reports:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix
- Positive prediction rate

ROC-AUC is calculated from predicted probabilities:

```python
roc_auc_score(y_test, predicted_probabilities)
```

rather than from hard class predictions.

### About the original notebook's ~94% XGBoost result

The original notebook reported approximately 94% accuracy/AUC for its XGBoost experiments.

Those numbers are **not copied here as final performance claims** because the original experiment used the test set for XGBoost early stopping/tuning and applied SMOTE before splitting.

The corrected training pipeline must be run before any new performance number is reported.

## Training

Install dependencies and regenerate the model:

```bash
uv sync
uv run python -m src.train
```

Training creates:

```text
model/model.bin
model/metrics.json
```

`metrics.json` contains:

- grouped-CV scores for every model family
- selected model
- selected hyperparameters
- validation threshold
- final untouched-test metrics
- train/validation/test patient counts

## API

Start the API after training:

```bash
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696
```

Endpoint:

```text
POST /predict
```

The API accepts the raw patient fields and runs the exact serialized feature-engineering pipeline used during training.

Example response:

```json
{
  "readmitted_probability": 0.23,
  "readmitted": true,
  "decision_threshold": 0.15,
  "model_name": "random_forest"
}
```

The values above are only an example of the response format.

## Docker

```bash
docker build -t patient-readmission-risk .
docker run -p 9696:9696 patient-readmission-risk
```

The image expects a regenerated `model/model.bin`.

## Repository structure

```text
data/
    diabetic_data.csv

notebooks/
    eda.ipynb
    logistic_regression.ipynb
    random_forest_xgboost.ipynb

src/
    preprocessing.py
    train.py
    predict.py

tests/
    test_pipeline.py

model/
    model.bin
    metrics.json

Dockerfile
pyproject.toml
uv.lock
```

The notebook remains useful for exploratory analysis and understanding the original experiments. The `src/` pipeline is the reproducible implementation used for training and deployment.

## Reproducibility and rollback

The exact repository state from before the corrective work is preserved on:

```text
backup/baseline-2026-10-02
```

The corrected work is developed on:

```text
fix/robust-readmission-pipeline
```

The final model artifact should be regenerated after the training code changes; existing legacy artifacts are not treated as validated results.
