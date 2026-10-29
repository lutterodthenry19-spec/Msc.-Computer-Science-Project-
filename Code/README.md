# Reproducible fraud-detection evaluation

The revised experiment compares class-weighted logistic regression, random forest, and XGBoost with the same classifiers trained on SMOTE-ENN-resampled data. It uses an ordered holdout from the ULB European Credit Card Fraud Detection dataset.

## Dataset

Obtain the original `creditcard.csv` file from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), then place it at `data/creditcard.csv` in the repository root. The source data is not stored in Git.

The source data has transaction time, anonymized PCA variables, amount, and a fraud label. It contains no stable cardholder identifier, so the experiment does not construct card-level sequences or claim to evaluate a GRU.

## Run

Use Python 3.12 and install the pinned dependencies:

```powershell
python -m pip install -r Code/requirements.txt
python Code/reviewer_safe_evaluation.py --data data/creditcard.csv --output results/reviewer_safe
```

The script sorts by `Time` and keeps equal timestamps together when creating chronological training, validation, and test partitions as close as possible to 70/15/15. It fits the scaler on training data, applies SMOTE-ENN to training data only, selects each classifier's F1 threshold on validation data, and evaluates the untouched test set once.

## Outputs

- `metrics.csv`: model settings, validation average precision, selected threshold, held-out metrics, confidence intervals, training time, CPU inference latency, and serialized model size.
- `paired_comparison.json`: class-stratified bootstrap interval for the random-forest F1 difference and exact paired test for fraud detection outcomes.
- `permutation_importance.csv`: validation-set permutation feature importance for the model selected by validation average precision.
- `run_metadata.json`: split boundaries and sizes, preprocessing and resampling rules, SMOTE-ENN preprocessing time, and runtime versions.
- `validation_probabilities.npz`: validation labels and probabilities for the two random-forest resampling comparisons.
- `figures/`: vector SVG and 600 dpi PNG versions of the evaluation workflow, validation precision-recall, and permutation-importance figures.

Confidence intervals use a stratified bootstrap that preserves the test partition's fraud and non-fraud counts. The reported serialized model size is a storage measure; it is not a measurement of peak resident memory on a target device. Inference latency is measured on the local CPU and should not be interpreted as a device-independent deployment guarantee.
