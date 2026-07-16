"""
Run all baseline models and compare results.
Does NOT modify any existing code.

Usage: python -m train.baseline_runner
"""
import numpy as np
from datetime import datetime
from logger import logger, setup_file_handler


def compute_metrics(preds, gts):
    signed = preds - gts
    sigma = float(np.std(signed))
    return {
        'log10_mae': float(np.mean(np.abs(signed))),
        'sigma': sigma,
        'mean_bias': float(np.mean(signed)),
        'within_1sigma': float(np.mean(np.abs(signed) < sigma) * 100.0),
        'within_2sigma': float(np.mean(np.abs(signed) < 2 * sigma) * 100.0),
    }


def run_all_baselines():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_handler(f"baselines_{timestamp}")

    results = {}

    # ── XGBoost ──
    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost baseline...")
    logger.info("=" * 60)
    from baseline_models.baseline_xgboost import train_xgboost
    xgb_res = train_xgboost()
    results['XGBoost'] = compute_metrics(xgb_res['preds'], xgb_res['gts'])

    # ── MLP ──
    logger.info("\n" + "=" * 60)
    logger.info("Training MLP baseline...")
    logger.info("=" * 60)
    from baseline_models.baseline_mlp import train_mlp
    mlp_res = train_mlp()
    results['MLP'] = compute_metrics(mlp_res['preds'], mlp_res['gts'])

    # ── LSTM ──
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM baseline...")
    logger.info("=" * 60)
    from baseline_models.baseline_lstm import train_lstm
    lstm_res = train_lstm()
    results['LSTM'] = compute_metrics(lstm_res['preds'], lstm_res['gts'])

    # ── Comparison Table ──
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 70)
    header = f"{'Model':<12} {'Log10-MAE ↓':>12} {'1σ ↓':>12} {'Mean Bias':>12} {'In 1σ %':>10} {'In 2σ %':>10}"
    logger.info(header)
    logger.info("-" * 70)
    for name in ['XGBoost', 'MLP', 'LSTM']:
        m = results[name]
        logger.info(
            f"{name:<12} {m['log10_mae']:>12.4f} {m['sigma']:>12.4f} "
            f"{m['mean_bias']:>+12.4f} {m['within_1sigma']:>9.1f}% "
            f"{m['within_2sigma']:>9.1f}%"
        )
    logger.info("=" * 70)

    # ── Generate scatter plots ──
    logger.info("\nGenerating scatter plots for all baselines...")
    from baseline_models.baseline_evaluate import plot_eval_results, compute_global_metrics
    for name in ['XGBoost', 'MLP', 'LSTM']:
        m = results[name]
        plot_eval_results(m['preds'], m['gts'],
                          compute_global_metrics(m['preds'], m['gts']),
                          name, timestamp)
    logger.info("All baseline evaluation plots saved to figures/")

    return results


if __name__ == "__main__":
    run_all_baselines()
