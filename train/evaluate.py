import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train.models import TCN
from train.dataset import SatelliteCollisionDataset
import os
from logger import logger

if not os.path.exists('logs'):
    os.makedirs('logs', exist_ok=True)

def format_breakdown_value(value, suffix=''):
    if value is None:
        return '-'
    return f"{value:.1f}{suffix}" if suffix else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def load_best_model(model_param, device, timestamp=''):
    model = TCN(**model_param)
    model.load_state_dict(
        torch.load(f'params/best_model_{timestamp}.pth', map_location=device, weights_only=True),
        strict=True)
    model.eval()
    model.to(device)
    return model


def run_inference(model, val_features, val_labels, device, seq_length=240):
    val_dataset = SatelliteCollisionDataset(val_features, val_labels, seq_length=seq_length)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)
    all_mix, all_low, all_high, all_gate_low, all_gts = [], [], [], [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            mixture, pred_low, pred_high, gate_logits = model(batch_features)
            gate_probs = torch.sigmoid(gate_logits)
            all_mix.extend(mixture.cpu().numpy().flatten())
            all_low.extend(pred_low.cpu().numpy().flatten())
            all_high.extend(pred_high.cpu().numpy().flatten())
            all_gate_low.extend(gate_probs[:, 0].cpu().numpy().flatten())
            all_gts.extend(batch_labels.numpy().flatten())
    return (np.array(all_mix), np.array(all_low), np.array(all_high),
            np.array(all_gate_low), np.array(all_gts))


def compute_global_metrics(log_preds, log_gts):
    signed = log_preds - log_gts
    errors = np.abs(signed)
    sigma  = float(np.std(signed))
    return {
        'mean_bias': float(np.mean(signed)),
        'sigma': sigma,
        'within_1sigma': float(np.mean(errors < sigma) * 100.0),
        'within_2sigma': float(np.mean(errors < 2 * sigma) * 100.0),
        'log10_mae': float(np.mean(errors)),
        'log10_median': float(np.median(errors)),
    }


def compute_breakdown(log_preds, log_gts, global_sigma):
    signed  = log_preds - log_gts
    errors  = np.abs(signed)
    decades = [(-8, -6), (-6, -5), (-5, -4), (-4, -3), (-3, -2), (-2, 0)]
    rows = []
    for low, high in decades:
        mask  = (log_gts >= low) & (log_gts < high)
        count = int(np.sum(mask))
        if count > 0:
            d_sigma = float(np.std(signed[mask]))
            row = {
                'pc_interval': f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count': count,
                'bias': float(np.mean(signed[mask])),
                'sigma':d_sigma,
                'within_1sigma': float(np.mean(errors[mask] < global_sigma) * 100.0),
                'within_2sigma': float(np.mean(errors[mask] < 2 * global_sigma) * 100.0),
            }
        else:
            row = {
                'pc_interval': f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count': 0,
                'bias': None,
                'sigma': None,
                'within_1sigma': None,
                'within_2sigma': None,
            }
        rows.append(row)
    return rows


def print_global_summary(metrics, n_samples):
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation on ALL {n_samples} validation samples")
    logger.info(f"{'='*60}")
    logger.info(f" Mean bias (μ): {metrics['mean_bias']:+.4f} orders of magnitude")
    logger.info(f" 1σ: {metrics['sigma']:.4f} orders of magnitude")
    logger.info(f" 2σ: {2*metrics['sigma']:.4f} orders of magnitude")
    logger.info(f" Within 1σ: {metrics['within_1sigma']:.1f}%  (ideal 68.3%)")
    logger.info(f" Within 2σ: {metrics['within_2sigma']:.1f}%  (ideal 95.4%)")
    logger.info(f" Log10-MAE: {metrics['log10_mae']:.4f} orders of magnitude")
    logger.info(f" Log10-Median: {metrics['log10_median']:.4f} orders of magnitude")


def print_breakdown(breakdown_rows):
    logger.info(f"\n{'─'*75}")
    logger.info(f"{'Pc Range':<18} {'Count':>6} {'Bias(μ)':>9} {'1σ':>8} {'In 1σ':>8} {'In 2σ':>8}")
    logger.info(f"{'─'*75}")
    for row in breakdown_rows:
        logger.info(
            f"{row['pc_interval']:<18} {row['count']:>6d} "
            f"{format_breakdown_value(row['bias']):>9} "
            f"{format_breakdown_value(row['sigma']):>8} "
            f"{format_breakdown_value(row['within_1sigma'], '%'):>8} "
            f"{format_breakdown_value(row['within_2sigma'], '%'):>8}"
        )



def plot_eval_results(log_preds, log_gts, metrics, breakdown_rows, timestamp=''):
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

    # Scatter: pred vs gt
    lims = [-8, 0]
    scatter = ax.scatter(log_gts, log_preds, c=log10_errors, cmap='RdYlGn_r',
                         alpha=0.6, s=12, vmin=0, vmax=3)
    ax.plot(lims, lims, 'r--', linewidth=1,   label='Perfect')
    ax.plot(lims, [l + sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5,
            label=f'±1σ ({sigma:.2f})')
    ax.plot(lims, [l - sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5)
    ax.plot(lims, [l + 2*sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3,
            label=f'±2σ ({2*sigma:.2f})')
    ax.plot(lims, [l - 2*sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground Truth log10(Pc)', fontsize=scatter_fontsize)
    ax.set_ylabel('Predicted log10(Pc)', fontsize=scatter_fontsize)
    ax.set_title(f'Prediction vs GT (1σ={sigma:.3f} orders)', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('log10 error', fontsize=scatter_fontsize)
    colorbar.ax.tick_params(labelsize=tick_fontsize)

    # Histogram: signed error distribution
    ax2.hist(log10_signed, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=sigma, color='b', linestyle='--', alpha=0.7, label=f'±1σ ({sigma:.3f})')
    ax2.axvline(x=-sigma, color='b', linestyle='--', alpha=0.7)
    ax2.axvline(x=2*sigma, color='g', linestyle='--', alpha=0.5, label=f'±2σ ({2*sigma:.3f})')
    ax2.axvline(x=-2*sigma, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=mean_bias, color='orange', linestyle='-',  linewidth=2,
                label=f'Bias={mean_bias:+.3f}')
    ax2.set_xlabel('Signed Log10 Error (orders of magnitude)', fontsize=scatter_fontsize)
    ax2.set_ylabel('Count', fontsize=scatter_fontsize)
    ax2.set_title('Error Distribution (Signed)', fontsize=title_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)
    ax2.grid(True, alpha=0.3)

    # Table: per-decade breakdown
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis('off')
    table_ax.set_title('Evaluation Set Group Performance Breakdown', fontsize=title_fontsize, pad=12)
    table_data = [
        [
            row['pc_interval'],
            str(row['count']),
            format_breakdown_value(row['bias']),
            format_breakdown_value(row['sigma']),
            format_breakdown_value(row['within_1sigma'], '%'),
            format_breakdown_value(row['within_2sigma'], '%'),
        ]
        for row in breakdown_rows
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=['Pc interval', 'Count', 'Bias', '1σ', 'Within 1σ', 'Within 2σ'],
        cellLoc='center',
        loc='center',
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
    plt.savefig(f'figures/prediction_scatter_{timestamp}.png', dpi=200)
    plt.close()
    logger.info("Evaluation plots saved to 'figures/' directory")


def evaluate_best_model(model_param, val_features, val_labels, device='cuda',
                        seq_length=240, timestamp='', moe_threshold=-3.5):
    model = load_best_model(model_param, device, timestamp=timestamp)
    log_preds, low_preds, high_preds, gate_low_prob, log_gts = run_inference(
        model, val_features, val_labels, device, seq_length)
    
    metrics = compute_global_metrics(log_preds, log_gts)
    breakdown_rows = compute_breakdown(log_preds, log_gts, metrics['sigma'])

    print_global_summary(metrics, len(log_gts))
    print_breakdown(breakdown_rows)

    mask_low = log_gts < moe_threshold
    mask_high = log_gts >= moe_threshold

    gate_low_region = float(np.mean(gate_low_prob[mask_low])) if mask_low.sum() > 0 else float('nan')
    gate_high_region = float(np.mean(1.0 - gate_low_prob[mask_high])) if mask_high.sum() > 0 else float('nan')
    low_exp_mae = float(np.mean(np.abs(low_preds[mask_low] - log_gts[mask_low]))) if mask_low.sum() > 0 else float('nan')
    high_exp_mae = float(np.mean(np.abs(high_preds[mask_high] - log_gts[mask_high]))) if mask_high.sum() > 0 else float('nan')

    logger.info(f"\n{'─'*50}")
    logger.info(f"Gate & Expert Diagnostics (threshold={moe_threshold})")
    logger.info(f"{'─'*50}")
    logger.info(f" Gate confidence (low region):  {gate_low_region:.4f}")
    logger.info(f" Gate confidence (high region): {gate_high_region:.4f}")
    logger.info(f" Expert_low MAE (low region):   {low_exp_mae:.4f} orders")
    logger.info(f" Expert_high MAE (high region): {high_exp_mae:.4f} orders")

    plot_eval_results(log_preds, log_gts, metrics, breakdown_rows, timestamp=timestamp)
