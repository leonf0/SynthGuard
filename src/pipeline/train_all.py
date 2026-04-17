from typing import Dict

from ..data import (
    build_sequence_dataset,
    build_training_dataset,
    fetch_equities_returns,
    fetch_vix_pipeline,
)
from ..generators import (
    train_cvae,
    train_diffusion,
    train_garch,
    train_gbm,
    train_heston,
    train_tft,
)
from ..regimes import (
    estimate_transition_matrix,
    fit_regime_gmm,
    make_markov_step,
    smooth_regime_labels,
)
from ..config import WINDOW


def train_all() -> Dict:
    print("=== Fetching VIX + TNX data ===")
    vix_raw, features_df, X_scaled, scaler, dates = fetch_vix_pipeline()

    print("=== Fetching equities returns ===")
    returns_df = fetch_equities_returns()

    print("=== Fitting GMM (K=3) ===")
    gmm, regime_labels_raw = fit_regime_gmm(X_scaled, features_df, dates)
    regime_labels = smooth_regime_labels(regime_labels_raw)
    print(
        f"  Regime distribution:\n"
        f"{regime_labels.value_counts(normalize=True).to_string()}"
    )

    print("=== Estimating Markov chain ===")
    trans_matrix = estimate_transition_matrix(regime_labels)
    markov_step = make_markov_step(trans_matrix)
    print(f"  Transition matrix:\n{trans_matrix}")

    print("=== Building pointwise training dataset (window={}) ===".format(WINDOW))
    X_windows, X_regimes, y_targets = build_training_dataset(returns_df, regime_labels)
    print(f"  Pointwise: {len(y_targets)} samples")

    print("=== Building sequence training dataset (len=252) for diffusion ===")
    sequences, seq_regimes = build_sequence_dataset(returns_df, regime_labels)
    print(f"  Sequence-level: {len(sequences)} sequences")

    print("=== Training GBM ===")
    gbm_params = train_gbm(returns_df, regime_labels)

    print("=== Training GARCH(1,1) ===")
    garch_params = train_garch(returns_df, regime_labels)

    print("=== Calibrating Heston ===")
    heston_params = train_heston(returns_df, regime_labels)

    print("=== Training TFT ===")
    tft_model, tft_residual = train_tft(X_windows, X_regimes, y_targets)

    print("=== Training CVAE ===")
    cvae_model, cvae_residual = train_cvae(X_windows, X_regimes, y_targets)

    print("=== Training Diffusion (full-sequence) ===")
    diff_model = train_diffusion(sequences, seq_regimes)

    print("=== All training complete ===")
    return {
        "gmm": gmm,
        "scaler": scaler,
        "regime_labels": regime_labels,
        "transition_matrix": trans_matrix,
        "markov_step": markov_step,
        "training_data": (X_windows, X_regimes, y_targets),
        "sequence_training_data": (sequences, seq_regimes),
        "gbm_params": gbm_params,
        "garch_params": garch_params,
        "heston_params": heston_params,
        "tft_model": tft_model,
        "tft_residual_std": tft_residual,
        "cvae_model": cvae_model,
        "cvae_residual_std": cvae_residual,
        "diffusion_model": diff_model,
    }
