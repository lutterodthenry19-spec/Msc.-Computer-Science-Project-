"""Leakage-aware benchmark for the ULB credit-card fraud dataset.

The dataset has no customer/card identifier, so this script treats each record
as an independent tabular observation and does not claim to model card-level
transaction sequences. It makes a chronological train/validation/test split,
fits preprocessing on training data only, applies SMOTE-ENN to training data
only, chooses thresholds on validation data, and evaluates the test set once.

Example:
    python reviewer_safe_evaluation.py --data ../data/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 42
TARGET = "Class"
TIME = "Time"
FEATURES = [TIME, *[f"V{i}" for i in range(1, 29)], "Amount"]


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Split on whole timestamp groups, nearest to 70/15/15 by row count."""
    if frame[TIME].isna().any():
        raise ValueError("Time contains missing values; resolve them before splitting.")

    group_counts = frame.groupby(TIME, sort=True).size()
    cumulative = group_counts.cumsum().to_numpy()
    timestamp_values = group_counts.index.to_numpy()
    n_rows = len(frame)

    first_idx = int(np.argmin(np.abs(cumulative - 0.70 * n_rows)))
    second_candidates = np.arange(first_idx + 1, len(cumulative))
    if len(second_candidates) == 0:
        raise ValueError("Unable to create three non-empty chronological splits.")
    second_idx = int(
        second_candidates[
            np.argmin(np.abs(cumulative[second_candidates] - 0.85 * n_rows))
        ]
    )
    first_time = timestamp_values[first_idx]
    second_time = timestamp_values[second_idx]

    train = frame.loc[frame[TIME] <= first_time].copy()
    validation = frame.loc[
        (frame[TIME] > first_time) & (frame[TIME] <= second_time)
    ].copy()
    test = frame.loc[frame[TIME] > second_time].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological splitting produced an empty partition.")
    if not (train[TIME].max() < validation[TIME].min()):
        raise AssertionError("Train and validation timestamps overlap.")
    if not (validation[TIME].max() < test[TIME].min()):
        raise AssertionError("Validation and test timestamps overlap.")
    return train, validation, test


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the F1-maximizing threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials**2))
        / denominator
    )
    return [float(max(0.0, centre - margin)), float(min(1.0, centre + margin))]


def stratified_bootstrap_f1_interval(
    y_true: np.ndarray,
    predictions: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    """Percentile interval resampling test fraud and non-fraud separately."""
    positive_predictions = predictions[y_true == 1]
    negative_predictions = predictions[y_true == 0]
    if len(positive_predictions) == 0 or len(negative_predictions) == 0:
        return [float("nan"), float("nan")]

    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    batch_size = 100
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        sampled_positive = positive_predictions[
            rng.integers(0, len(positive_predictions), size=(size, len(positive_predictions)))
        ].sum(axis=1)
        sampled_false_positive = negative_predictions[
            rng.integers(0, len(negative_predictions), size=(size, len(negative_predictions)))
        ].sum(axis=1)
        false_negative = len(positive_predictions) - sampled_positive
        f1 = np.divide(
            2 * sampled_positive,
            2 * sampled_positive + sampled_false_positive + false_negative,
            out=np.zeros(size, dtype=float),
            where=(2 * sampled_positive + sampled_false_positive + false_negative) > 0,
        )
        values.append(f1)
    bounds = np.percentile(np.concatenate(values), [2.5, 97.5])
    return [float(bounds[0]), float(bounds[1])]


def paired_f1_bootstrap_delta(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    """Paired, class-stratified bootstrap interval for second minus first F1."""
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    for start in range(0, repetitions, 100):
        size = min(100, repetitions - start)
        sampled = np.concatenate(
            [
                rng.choice(pos, size=(size, len(pos)), replace=True),
                rng.choice(neg, size=(size, len(neg)), replace=True),
            ],
            axis=1,
        )
        labels = y_true[sampled]
        pred_a = first[sampled]
        pred_b = second[sampled]
        tp_a = np.sum((labels == 1) & (pred_a == 1), axis=1)
        fp_a = np.sum((labels == 0) & (pred_a == 1), axis=1)
        fn_a = np.sum((labels == 1) & (pred_a == 0), axis=1)
        tp_b = np.sum((labels == 1) & (pred_b == 1), axis=1)
        fp_b = np.sum((labels == 0) & (pred_b == 1), axis=1)
        fn_b = np.sum((labels == 1) & (pred_b == 0), axis=1)
        f1_a = np.divide(2 * tp_a, 2 * tp_a + fp_a + fn_a)
        f1_b = np.divide(2 * tp_b, 2 * tp_b + fp_b + fn_b)
        deltas.append(f1_b - f1_a)
    bounds = np.percentile(np.concatenate(deltas), [2.5, 97.5])
    return [float(bounds[0]), float(bounds[1])]


def score_at_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float, repetitions: int
) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision_ci = wilson_interval(int(tp), int(tp + fp))
    recall_ci = wilson_interval(int(tp), int(tp + fn))
    f1_ci = stratified_bootstrap_f1_interval(y_true, predictions, repetitions, SEED)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    scores = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "precision_ci_low": precision_ci[0],
        "precision_ci_high": precision_ci[1],
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f1_ci_low": f1_ci[0],
        "f1_ci_high": f1_ci[1],
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return scores, predictions


def inference_latency_ms(model, x: np.ndarray) -> float:
    sample = x[: min(5000, len(x))]
    model.predict_proba(sample)
    timings = []
    for _ in range(7):
        start = time.perf_counter()
        model.predict_proba(sample)
        timings.append((time.perf_counter() - start) * 1000 / len(sample))
    return float(np.median(timings))


def model_definitions(negative: int, positive: int) -> dict[str, object]:
    weight = negative / positive
    return {
        "Logistic regression | class weight": LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=SEED
        ),
        "Random forest | class weight": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost | class weight": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=weight,
            max_delta_step=1,
            eval_metric="aucpr",
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=SEED,
        ),
    }


def save_figures(
    output_dir: Path,
    y_validation: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    importances: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    prevalence = float(np.mean(y_validation))
    ax.axhline(prevalence, color="0.45", linestyle="--", label=f"Prevalence ({prevalence:.4f})")
    for name, scores in probabilities_by_model.items():
        precision, recall, _ = precision_recall_curve(y_validation, scores)
        ax.plot(recall, precision, linewidth=1.6, label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_pr_curves.svg", format="svg")
    fig.savefig(figure_dir / "validation_pr_curves.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    top = importances.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3569a8")
    ax.set_xlabel("Decrease in validation average precision after permutation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_permutation_importance.svg", format="svg")
    fig.savefig(figure_dir / "validation_permutation_importance.png", dpi=600)
    plt.close(fig)


def save_workflow_figure(output_dir: Path) -> None:
    """Draw the leakage controls as a concise, publication-quality diagram."""
    from matplotlib.patches import FancyBboxPatch

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    boxes = [
        (0.65, 5.55, 4.9, 0.82, "Original records sorted by transaction Time"),
        (0.65, 4.30, 4.9, 0.82, "Chronological split on whole Time groups\n70% training / 15% validation / 15% test"),
        (0.65, 2.90, 4.9, 1.00, "Training partition only\nFit the scaler and models; apply SMOTE-ENN to training variants"),
        (0.65, 1.50, 4.9, 0.82, "Validation partition\nSelect each model's F1 threshold"),
        (0.65, 0.20, 4.9, 0.82, "Untouched test partition\nFinal metrics and confidence intervals"),
    ]
    for x, y, width, height, label in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#26384a",
            facecolor="#eef3f8",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9.2)
    for y1, y2 in [(5.55, 5.12), (4.30, 3.90), (2.90, 2.32), (1.50, 1.02)]:
        ax.annotate(
            "", xy=(3.10, y2), xytext=(3.10, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#26384a"},
        )
    fig.tight_layout(pad=0.25)
    fig.savefig(figure_dir / "evaluation_workflow.svg", format="svg", bbox_inches="tight")
    fig.savefig(figure_dir / "evaluation_workflow.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    frame = pd.read_csv(data_path)
    required = set([*FEATURES, TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Required dataset columns contain missing values.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Class must contain exactly the binary labels 0 and 1.")

    frame = frame.sort_values(TIME, kind="mergesort").reset_index(drop=True)
    train, validation, test = chronological_split(frame)

    x_train_raw = train[FEATURES].to_numpy(dtype=np.float64)
    x_validation_raw = validation[FEATURES].to_numpy(dtype=np.float64)
    x_test_raw = test[FEATURES].to_numpy(dtype=np.float64)
    y_train = train[TARGET].to_numpy(dtype=np.uint8)
    y_validation = validation[TARGET].to_numpy(dtype=np.uint8)
    y_test = test[TARGET].to_numpy(dtype=np.uint8)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_validation = scaler.transform(x_validation_raw)
    x_test = scaler.transform(x_test_raw)

    smote_enn = SMOTEENN(
        smote=SMOTE(k_neighbors=5, random_state=SEED),
        enn=EditedNearestNeighbours(n_neighbors=3),
        random_state=SEED,
    )
    resampling_started = time.perf_counter()
    x_train_resampled, y_train_resampled = smote_enn.fit_resample(x_train, y_train)
    smote_enn_fit_resample_seconds = float(time.perf_counter() - resampling_started)
    models = model_definitions(int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))

    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, object] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []

    for name, base_model in models.items():
        variants = [(name, base_model, x_train, y_train)]
        resampled_name = name.replace(" | class weight", " | SMOTE-ENN")
        if "Logistic regression" in name:
            resampled_model = LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=SEED
            )
        elif "Random forest" in name:
            resampled_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )
        else:
            resampled_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=float(np.sum(y_train_resampled == 0) / np.sum(y_train_resampled == 1)),
                max_delta_step=1,
                eval_metric="aucpr",
                tree_method="hist",
                device="cpu",
                n_jobs=4,
                random_state=SEED,
            )
        variants.append((resampled_name, resampled_model, x_train_resampled, y_train_resampled))

        for variant_name, model, x_fit, y_fit in variants:
            start = time.perf_counter()
            model.fit(x_fit, y_fit)
            training_seconds = float(time.perf_counter() - start)
            fitted_models[variant_name] = model

            validation_probabilities[variant_name] = model.predict_proba(x_validation)[:, 1]
            test_probabilities[variant_name] = model.predict_proba(x_test)[:, 1]
            threshold = select_threshold(y_validation, validation_probabilities[variant_name])
            test_scores, y_prediction = score_at_threshold(
                y_test, test_probabilities[variant_name], threshold, args.bootstrap
            )
            predictions[variant_name] = y_prediction
            records.append(
                {
                    "model": variant_name,
                    "train_rows_before_resampling": int(len(y_train)),
                    "train_rows_used": int(len(y_fit)),
                    "training_seconds": training_seconds,
                    "validation_average_precision": float(
                        average_precision_score(y_validation, validation_probabilities[variant_name])
                    ),
                    "test_threshold_selected_on_validation": threshold,
                    "test_inference_ms_per_transaction": inference_latency_ms(model, x_test),
                    "serialized_model_bytes": int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))),
                    **test_scores,
                }
            )
            print(
                f"{variant_name}: validation AP="
                f"{records[-1]['validation_average_precision']:.4f}; "
                f"test F1={test_scores['f1']:.4f} "
                f"[{test_scores['f1_ci_low']:.4f}, {test_scores['f1_ci_high']:.4f}]"
            )

    results = pd.DataFrame(records)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "validation_probabilities.npz",
        y_validation=y_validation,
        **{f"model_{i}": values for i, values in enumerate(validation_probabilities.values())},
    )

    # Select the model for the explanation by validation average precision only.
    selected_name = str(results.loc[results["validation_average_precision"].idxmax(), "model"])
    selected_model_obj = fitted_models[selected_name]

    importance = permutation_importance(
        selected_model_obj,
        x_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)

    # The exploratory resampling comparison is paired on the same test records.
    rf_weighted = "Random forest | class weight"
    rf_resampled = "Random forest | SMOTE-ENN"
    pos = y_test == 1
    first_detects = predictions[rf_weighted][pos] == 1
    second_detects = predictions[rf_resampled][pos] == 1
    first_only = int(np.sum(first_detects & ~second_detects))
    second_only = int(np.sum(~first_detects & second_detects))
    mcnemar_p = (
        float(binomtest(min(first_only, second_only), first_only + second_only, 0.5).pvalue)
        if first_only + second_only
        else 1.0
    )
    comparison = {
        "paired_comparison": f"{rf_resampled} minus {rf_weighted}",
        "test_f1_difference_ci95": paired_f1_bootstrap_delta(
            y_test, predictions[rf_weighted], predictions[rf_resampled], args.bootstrap, SEED
        ),
        "fraud_cases_detected_only_by_class_weighted_rf": first_only,
        "fraud_cases_detected_only_by_smote_enn_rf": second_only,
        "exact_mcnemar_p_for_paired_fraud_detection": mcnemar_p,
        "note": "Paired comparison is restricted to the fraud cases in the untouched chronological test partition.",
    }
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    save_figures(output_dir, y_validation, validation_probabilities, importance_frame)

    split_summary = {}
    for label, part in [("train", train), ("validation", validation), ("test", test)]:
        split_summary[label] = {
            "rows": int(len(part)),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "time_min": float(part[TIME].min()),
            "time_max": float(part[TIME].max()),
        }
    metadata = {
        "dataset": "ULB European Credit Card Fraud Detection (Kaggle mlg-ulb/creditcardfraud)",
        "rows": int(len(frame)),
        "features": FEATURES,
        "target": TARGET,
        "split_rule": "whole Time groups; closest attainable 70/15/15 chronological proportions",
        "scaler": "StandardScaler fit on training only; applied unchanged to validation and test",
        "resampling": "SMOTE-ENN fit on training partition only; validation and test untouched",
        "threshold_rule": "maximize validation F1; test evaluated at that fixed threshold",
        "bootstrap_repetitions": int(args.bootstrap),
        "selected_explanation_model_by_validation_average_precision": selected_name,
        "split_summary": split_summary,
        "resampled_training_rows": int(len(y_train_resampled)),
        "resampled_training_frauds": int(np.sum(y_train_resampled == 1)),
        "resampled_training_legitimate": int(np.sum(y_train_resampled == 0)),
        "smote_enn_fit_resample_seconds": smote_enn_fit_resample_seconds,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": __import__("imblearn").__version__,
            "xgboost": xgboost.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_workflow_figure(output_dir)
    print(f"\nResults written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the original creditcard.csv file")
    parser.add_argument("--output", default="../results/reviewer_safe", help="Output directory")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Stratified bootstrap repetitions")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
