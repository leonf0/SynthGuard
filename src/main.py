import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import fetch_individual_returns
from .discriminator import predict_tcn, train_tcn
from .pipeline import build_discriminator_dataset, train_all


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=== Stage 1: train all generators / regime machinery ===")
    train_all_dict = train_all()

    print("=== Stage 2: fetch individual returns ===")
    individual_returns = fetch_individual_returns()

    print("=== Stage 3: build discriminator dataset ===")
    X_returns, X_regimes, y = build_discriminator_dataset(
        train_all_dict=train_all_dict,
        individual_returns=individual_returns,
    )
    print(f"X_returns shape: {X_returns.shape}")
    print(f"X_regimes shape: {X_regimes.shape}")
    print(f"y shape: {y.shape}")
    print(f"Synthetic fraction: {y.mean():.2%}")

    n = len(y)
    n_test = int(0.20 * n)

    X_returns_train = X_returns[:-n_test]
    X_regimes_train = X_regimes[:-n_test]
    y_train = y[:-n_test]

    X_returns_test = X_returns[-n_test:]
    X_regimes_test = X_regimes[-n_test:]
    y_test = y[-n_test:]

    print("=== Stage 4: train TCN on train split ===")
    tcn_dict = train_tcn(
        X_returns=X_returns_train,
        X_regimes=X_regimes_train,
        y=y_train,
        n_ensemble=5,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        patience=5,
        train_frac=0.80,
        val_frac=0.10,
    )

    print("=== Stage 5: test TCN on held-out split ===")
    y_pred, y_prob = predict_tcn(
        X_returns=X_returns_test,
        X_regimes=X_regimes_test,
        tcn_dict=tcn_dict,
        batch_size=512,
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return {
        "train_all_dict": train_all_dict,
        "individual_returns": individual_returns,
        "X_returns_train": X_returns_train,
        "X_regimes_train": X_regimes_train,
        "y_train": y_train,
        "X_returns_test": X_returns_test,
        "X_regimes_test": X_regimes_test,
        "y_test": y_test,
        "tcn_dict": tcn_dict,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
        },
    }


if __name__ == "__main__":
    results = main()
