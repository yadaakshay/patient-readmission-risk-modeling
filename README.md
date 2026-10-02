# Patient Readmission Risk Modeling

An end-to-end machine-learning project for predicting whether a diabetic
patient will be **readmitted within 30 days** using hospital encounter data.

The project started as a notebook-based modeling study and has been rebuilt
as a reproducible, leakage-safe training and deployment pipeline.

## Target

The target is:

- `1`: `readmitted == "<30"` — readmitted within 30 days
- `0`: `readmitted == ">30"` or `"NO"`

The prediction target is intentionally narrower than "any future
readmission."

## What the original notebook did

The original notebook explored:

- feature engineering
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- SMOTE
- 10-fold cross-validation
- model metrics and feature importance

Those ideas are retained, but the experimental design has been corrected.

### Problems found in the original notebook

The notebook applied SMOTE to the complete dataset before creating its
train/test split. It also discarded `patient_nbr` before splitting, so
different encounters from the same patient could cross the split boundary.

The XGBoost experiments repeatedly used the test set for early stopping and
hyperparameter exploration. In addition, ROC-AUC was sometimes calculated
from hard 0/1 predictions instead of probability scores.

Those choices can produce optimistic evaluation.

The repository therefore does **not** copy the notebook's reported ~94%
XGBoost result as a final performance claim. The model must be retrained
under the corrected protocol.

## Corrected modeling pipeline

The current training methodology is:

```
Raw encounters
      |
      v
Patient-level 60/20/20 split
      |
      +------------------------------+
      |                              |
      v                              v
Training patients              Untouched test patients
      |
      v
10-fold StratifiedGroupKFold
      |
      +--> fold training patients
      |       |
      |       v
      |   feature engineering
      |       |
      |       v
      |   one-hot encoding
      |       |
      |       v
      |      SMOTE
      |       |
      |       v
      |      model
      |
      +--> fold validation patients
              |
              v
        evaluation only

Training CV selects the model.
Validation patients select the classification threshold.
Only then is train + validation used for the final fit.
The test set is evaluated once at the end.
```

### Patient-level splitting

The source dataset contains multiple encounters for some patients.

`patient_nbr` is therefore used only as a grouping variable:

- it is never a model feature
- no patient appears in multiple outer splits
- no patient appears in multiple CV folds

This prevents the model from seeing one patient's encounters during training
and another encounter from the same patient during evaluation.

### SMOTE

SMOTE is implemented **inside the imbalanced-learn pipeline**:

```
feature engineering
      ↓
DictVectorizer
      ↓
SMOTE
      ↓
classifier
```

Therefore, during cross-validation, synthetic minority observations are
created only from the training portion of each fold.

SMOTE is never fitted on:

- the CV validation fold
- the separate validation set
- the final test set

This avoids resampling leakage.

> Note: standard SMOTE is being applied after one-hot encoding. That can
> create fractional values in encoded categorical dimensions. A future
> experiment can compare this with a categorical-aware method such as
> SMOTENC.

### Cross-validation

The project uses **10-fold StratifiedGroupKFold**.

- **Group:** `patient_nbr`
- **Stratification:** preserves the class distribution as well as possible
- **Scoring:** ROC-AUC

Four model families from the notebook are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The selected production model is the model with the highest mean
patient-grouped CV ROC-AUC.

The CV results are saved to:

```
model/metrics.json
```

No model is selected using the final test set.

## Feature engineering

The shared transformer in `src/preprocessing.py` incorporates the useful
feature engineering from the original notebook:

### Hospital utilization

- `patient_service`
- `total_previous_visits`
- `had_previous_inpatient`

### Medication behavior

- `num_med`
- `med_change`
- `num_medications_used`
- `num_adjusted_medications`
- `any_medication_change`
- `on_insulin`

### Treatment intensity

- `avg_medications_per_day`
- `procedure_to_lab_ratio`

### Diagnosis grouping

ICD-9 diagnosis codes are grouped into the same clinical categories used
in the notebook for primary, secondary, and additional diagnoses.

### Skew handling

The notebook identified several highly skewed count variables and applied
log transforms. The production transformer uses a fixed list of those
transforms instead of calculating skewness on the complete dataset.

That distinction matters: estimating transformation choices from the full
dataset before splitting can leak information across the evaluation
boundary.

### Other preprocessing

The pipeline also:

- normalizes categorical values
- maps age ranges to their midpoint
- groups admission/discharge/source categories
- handles missing values
- preserves the same transformations during API inference

## Model evaluation

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
roc_auc_score(y_test, probability)
```

rather than from thresholded 0/1 predictions.

### Classification threshold

The model outputs:

```
P(readmitted within 30 days | patient data)
```

A 0.5 threshold is not automatically appropriate for an imbalanced
classification problem.

The classification threshold is selected using the separate validation
set by maximizing F1 over thresholds from 0.05 to 0.50.

The test set is never used for threshold selection.

## Reproducible training

From the repository root:

```bash
uv sync
uv run python -m src.train
```

Training will:

1. load the raw dataset
2. create the target
3. make patient-disjoint train/validation/test splits
4. run 10-fold patient-grouped CV
5. apply SMOTE inside each CV training fold
6. compare the four model families
7. select the model using mean CV ROC-AUC
8. select the decision threshold on validation data
9. refit the selected pipeline on train + validation
10. evaluate once on the untouched test set
11. save the complete inference pipeline and threshold

Outputs:

```
model/model.bin
model/metrics.json
```

The currently committed `model/model.bin` is a legacy artifact from the
previous implementation and should be regenerated with the command above
before deployment.

## API

After training:

```bash
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696
```

POST to:

```
/predict
```

The response contains:

```json
{
  "readmitted_probability": 0.23,
  "readmitted": true,
  "decision_threshold": 0.17,
  "model_name": "random_forest"
}
```

The numbers are illustrative only.

The API loads the exact serialized preprocessing + model pipeline used
during training, preventing training-serving preprocessing skew.

## Docker

Build:

```bash
docker build -t patient-readmission-risk .
```

Run:

```bash
docker run -p 9696:9696 patient-readmission-risk
```

Regenerate `model/model.bin` before deploying the image.

## Repository structure

```
data/
    diabetic_data.csv

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

The original notebook remains the reference for exploratory analysis.
The `src/` pipeline is the reproducible implementation.

## Reproducibility and rollback

The exact pre-correction repository state is preserved on:

```
backup/baseline-2026-10-02
```

The corrected implementation is developed on:

```
fix/robust-readmission-pipeline
```

The original `main` branch is not modified by this work.
