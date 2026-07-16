"""
Evaluate saved baseline models and generate scatter plots (pred vs gt).
Matches the TCN evaluation plot style from evaluate.py.
Does NOT modify any existing code.

Usage:
    python -m train.baseline_evaluate
    python -m train.baseline_evaluate --xgboost params/xgb_baseline_20260709_171237.pkl
    python -m train.baseline_evaluate --mlp params/mlp_baseline_20260709_171355.pth
    python -m train.baseline_evaluate --lstm params/lstm_baseline_20260709_171355.pth
    python -m train.baseline_evaluate --all  # auto-discover all saved baselines
"""
import os
import json
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from logger import logger

from baseline_models.baseline_dataloader import load_baseline_data


# ── Plotting helpers (matching evaluate.py style) ──

def format_breakdown_value(value, suffix=''):
    if value is None:
        return '-'
    return f"{value:.1f}{suffix}" if suffix else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def compute_global_metrics(log_preds, log_gts):
    signed = log_preds - log_gts
    errors = np.abs(signed)
    sigma = float(np.std(signed))
    return {
        'mean_bias': float(np.mean(signed)),
        'sigma': sigma,
        'within_1sigma': float(np.mean(errors < sigma) * 100.0),
        'within_2sigma': float(np.mean(errors < 2 * sigma) * 100.0),
        'log10_mae': float(np.mean(errors)),
        'log10_median': float(np.median(errors)),
    }


def compute_breakdown(log_preds, log_gts, global_sigma):
    signed = log_preds - log_gts
    errors = np.abs(signed)
    decades = [(-8, -6), (-6, -5), (-5, -4), (-4, -3), (-3, -2), (-2, 0)]
    rows = []
    for low, high in decades:
        mask = (log_gts >= low) & (log_gts < high)
        count = int(np.sum(mask))
        if count > 0:
            d_sigma = float(np.std(signed[mask]))
            row = {
                'pc_interval': f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count': count,
                'bias': float(np.mean(signed[mask])),
                'sigma': d_sigma,
                'within_1sigma': float(np.mean(errors[mask] < global_sigma) * 100.0),
                'within_2sigma': float(np.mean(errors[mask] < 2 * global_sigma) * 100.0),
            }
        else:
            row = {
                'pc_interval': f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count': 0, 'bias': None, 'sigma': None,
                'within_1sigma': None, 'within_2sigma': None,
            }
        rows.append(row)
    return rows


def plot_eval_results(log_preds, log_gts, metrics, model_name, timestamp=''):
    log10_errors = np.abs(log_preds - log_gts)
    log10_signed = log_preds - log_gts
    sigma = metrics['sigma']
    mean_bias = metrics['mean_bias']
    scatter_fontsize = 14
    title_fontsize = 16
    legend_fontsize = 11
    tick_fontsize = 12

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2.2])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    lims = [-8, 0]
    scatter = ax.scatter(log_gts, log_preds, c=log10_errors, cmap='RdYlGn_r',
                         alpha=0.6, s=12, vmin=0, vmax=3)
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax.plot(lims, [l + sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5,
            label=f'±1σ ({sigma:.2f})')
    ax.plot(lims, [l - sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5)
    ax.plot(lims, [l + 2 * sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3,
            label=f'±2σ ({2 * sigma:.2f})')
    ax.plot(lims, [l - 2 * sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground Truth log10(Pc)', fontsize=scatter_fontsize)
    ax.set_ylabel('Predicted log10(Pc)', fontsize=scatter_fontsize)
    ax.set_title(f'{model_name}: Prediction vs GT (1σ={sigma:.3f} orders)', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('log10 error', fontsize=scatter_fontsize)
    colorbar.ax.tick_params(labelsize=tick_fontsize)

    ax2.hist(log10_signed, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=sigma, color='b', linestyle='--', alpha=0.7, label=f'±1σ ({sigma:.3f})')
    ax2.axvline(x=-sigma, color='b', linestyle='--', alpha=0.7)
    ax2.axvline(x=2 * sigma, color='g', linestyle='--', alpha=0.5, label=f'±2σ ({2 * sigma:.3f})')
    ax2.axvline(x=-2 * sigma, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=mean_bias, color='orange', linestyle='-', linewidth=2,
                label=f'Bias={mean_bias:+.3f}')
    ax2.set_xlabel('Signed Log10 Error (orders of magnitude)', fontsize=scatter_fontsize)
    ax2.set_ylabel('Count', fontsize=scatter_fontsize)
    ax2.set_title(f'{model_name}: Error Distribution (Signed)', fontsize=title_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)
    ax2.grid(True, alpha=0.3)

    # Table
    breakdown_rows = compute_breakdown(log_preds, log_gts, sigma)
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis('off')
    table_ax.set_title(f'{model_name}: Per-Decade Breakdown', fontsize=title_fontsize, pad=12)
    table_data = [
        [row['pc_interval'], str(row['count']),
         format_breakdown_value(row['bias']), format_breakdown_value(row['sigma']),
         format_breakdown_value(row['within_1sigma'], '%'),
         format_breakdown_value(row['within_2sigma'], '%')]
        for row in breakdown_rows
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=['Pc interval', 'Count', 'Bias', '1σ', 'Within 1σ', 'Within 2σ'],
        cellLoc='center', loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9e6f2')
        elif r % 2 == 1:
            cell.set_facecolor('#f7f7f7')

    plt.tight_layout()
    out_path = f'figures/{model_name.lower()}_scatter_{timestamp}.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Plot saved to {out_path}")


# ── Model loading and inference ──

def evaluate_xgboost(model_path, X_val, y_val, timestamp=''):
    logger.info(f"\n=== Evaluating XGBoost: {model_path} ===")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(X_val)
    metrics = compute_global_metrics(preds, y_val)
    log_metrics(metrics, 'XGBoost')
    plot_eval_results(preds, y_val, metrics, 'XGBoost', timestamp)
    return metrics


def evaluate_mlp(model_path, X_val, y_val, timestamp=''):
    logger.info(f"\n=== Evaluating MLP: {model_path} ===")
    from baseline_models.baseline_mlp import MLP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(input_dim=X_val.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_val).to(device)).cpu().numpy().flatten()
    metrics = compute_global_metrics(preds, y_val)
    log_metrics(metrics, 'MLP')
    plot_eval_results(preds, y_val, metrics, 'MLP', timestamp)
    return metrics


def evaluate_lstm(model_path, X_seq_val, y_val, timestamp=''):
    logger.info(f"\n=== Evaluating LSTM: {model_path} ===")
    from baseline_models.baseline_lstm import LSTMRegressor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMRegressor(input_dim=X_seq_val.shape[2])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Batch inference to avoid OOM on full validation set
    batch_size = 64
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_seq_val), batch_size):
            batch = torch.FloatTensor(X_seq_val[i:i + batch_size]).to(device)
            all_preds.append(model(batch).cpu().numpy().flatten())
    preds = np.concatenate(all_preds)

    metrics = compute_global_metrics(preds, y_val)
    log_metrics(metrics, 'LSTM')
    plot_eval_results(preds, y_val, metrics, 'LSTM', timestamp)
    return metrics


def log_metrics(metrics, name):
    logger.info(f"  {name} — Log10-MAE: {metrics['log10_mae']:.4f}  "
                f"1σ: {metrics['sigma']:.4f}  Bias: {metrics['mean_bias']:+.4f}  "
                f"1σ%: {metrics['within_1sigma']:.1f}%  2σ%: {metrics['within_2sigma']:.1f}%")


# ── Main ──

def discover_models():
    """Auto-discover saved baseline models in params/"""
    models = {}
    xgb_files = sorted(glob.glob('params/xgb_baseline_*.pkl'))
    mlp_files = sorted(glob.glob('params/mlp_baseline_*.pth'))
    lstm_files = sorted(glob.glob('params/lstm_baseline_*.pth'))
    if xgb_files:
        models['xgboost'] = xgb_files[-1]  # latest
    if mlp_files:
        models['mlp'] = mlp_files[-1]
    if lstm_files:
        models['lstm'] = lstm_files[-1]
    return models


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models with scatter plots')
    parser.add_argument('--xgboost', type=str, default=None, help='Path to XGBoost model (.pkl)')
    parser.add_argument('--mlp', type=str, default=None, help='Path to MLP model (.pth)')
    parser.add_argument('--lstm', type=str, default=None, help='Path to LSTM model (.pth)')
    parser.add_argument('--all', action='store_true', help='Auto-discover and evaluate all saved baselines')
    args = parser.parse_args()

    cfg = json.load(open('config.json'))
    data_cfg = cfg['data']

    # Load data
    X_flat_val, X_seq_val, y_val = load_baseline_data(
        data_cfg['feature_path'], data_cfg['label_path'],
        test_size=data_cfg['test_size'], random_state=data_cfg['random_state'],
        use_last_timestep=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.all:
        models = discover_models()
        if not models:
            logger.error("No saved baseline models found in params/")
            return
        logger.info(f"Discovered models: {list(models.keys())}")
    else:
        models = {}
        if args.xgboost:
            models['xgboost'] = args.xgboost
        if args.mlp:
            models['mlp'] = args.mlp
        if args.lstm:
            models['lstm'] = args.lstm

    if not models:
        logger.info("No models specified. Use --all or --xgboost/--mlp/--lstm.")
        logger.info("Auto-discovering...")
        models = discover_models()
        if not models:
            logger.error("No saved baseline models found in params/")
            return

    all_metrics = {}

    if 'xgboost' in models:
        m = evaluate_xgboost(models['xgboost'], X_flat_val, y_val, timestamp)
        all_metrics['XGBoost'] = m

    if 'mlp' in models:
        m = evaluate_mlp(models['mlp'], X_flat_val, y_val, timestamp)
        all_metrics['MLP'] = m

    if 'lstm' in models:
        m = evaluate_lstm(models['lstm'], X_seq_val, y_val, timestamp)
        all_metrics['LSTM'] = m

    # Summary comparison
    if len(all_metrics) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE MODEL COMPARISON (Validation Set)")
        logger.info("=" * 70)
        header = f"{'Model':<12} {'Log10-MAE ↓':>12} {'1σ ↓':>12} {'Mean Bias':>12} {'In 1σ %':>10} {'In 2σ %':>10}"
        logger.info(header)
        logger.info("-" * 70)
        for name, m in all_metrics.items():
            logger.info(
                f"{name:<12} {m['log10_mae']:>12.4f} {m['sigma']:>12.4f} "
                f"{m['mean_bias']:>+12.4f} {m['within_1sigma']:>9.1f}% "
                f"{m['within_2sigma']:>9.1f}%"
            )
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
