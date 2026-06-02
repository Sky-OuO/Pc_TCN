import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train.models import TCN
from train.dataset import SatelliteCollisionDataset
import os
import sys
from datetime import datetime

if not os.path.exists('logs'):
    os.makedirs('logs', exist_ok=True)

def format_breakdown_value(value, suffix=''):
    if value is None:
        return '-'
    return f"{value:.1f}{suffix}" if suffix else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def load_best_model(model_param, device):
    model = TCN(**model_param)
    model.load_state_dict(torch.load('params/best_model.pth', map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def run_inference(model, val_features, val_labels, device):
    val_dataset = SatelliteCollisionDataset(val_features, val_labels)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_prob = model(batch_features)
            all_preds.extend((10.0 ** log_prob).cpu().numpy().flatten())
            all_gts.extend(batch_labels.numpy().flatten())
    return np.array(all_preds), np.array(all_gts)


def compute_global_metrics(log_preds, log_gts):
    signed = log_preds - log_gts
    errors = np.abs(signed)
    sigma  = float(np.std(signed))
    return {
        'mean_bias':     float(np.mean(signed)),
        'sigma':         sigma,
        'within_1sigma': float(np.mean(errors < sigma) * 100.0),
        'within_2sigma': float(np.mean(errors < 2 * sigma) * 100.0),
        'log10_mae':     float(np.mean(errors)),
        'log10_median':  float(np.median(errors)),
    }


def compute_breakdown(log_preds, log_gts, global_sigma):
    signed  = log_preds - log_gts
    errors  = np.abs(signed)
    decades = [(-10, -8), (-8, -6), (-6, -5), (-5, -4), (-4, -3), (-3, -2), (-2, 0)]
    rows = []
    for low, high in decades:
        mask  = (log_gts >= low) & (log_gts < high)
        count = int(np.sum(mask))
        if count > 0:
            d_sigma = float(np.std(signed[mask]))
            row = {
                'pc_interval':   f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count':         count,
                'bias':          float(np.mean(signed[mask])),
                'sigma':         d_sigma,
                'within_1sigma': float(np.mean(errors[mask] < global_sigma) * 100.0),
                'within_2sigma': float(np.mean(errors[mask] < 2 * global_sigma) * 100.0),
            }
        else:
            row = {
                'pc_interval':   f"[1e{low:+.0f}, 1e{high:+.0f})",
                'count':         0,
                'bias':          None,
                'sigma':         None,
                'within_1sigma': None,
                'within_2sigma': None,
            }
        rows.append(row)
    return rows


def print_global_summary(metrics, n_samples):
    print(f"\n{'='*60}")
    print(f"Evaluation on ALL {n_samples} validation samples")
    print(f"{'='*60}")
    print(f"  Mean bias (μ):   {metrics['mean_bias']:+.4f} orders of magnitude")
    print(f"  1σ:              {metrics['sigma']:.4f} orders of magnitude")
    print(f"  2σ:              {2*metrics['sigma']:.4f} orders of magnitude")
    print(f"  Within 1σ:       {metrics['within_1sigma']:.1f}%  (ideal 68.3%)")
    print(f"  Within 2σ:       {metrics['within_2sigma']:.1f}%  (ideal 95.4%)")
    print(f"  Log10-MAE:       {metrics['log10_mae']:.4f} orders of magnitude")
    print(f"  Log10-Median:    {metrics['log10_median']:.4f} orders of magnitude")


def print_breakdown(breakdown_rows):
    print(f"\n{'─'*75}")
    print(f"{'Pc Range':<18} {'Count':>6} {'Bias(μ)':>9} {'1σ':>8} {'In 1σ':>8} {'In 2σ':>8}")
    print(f"{'─'*75}")
    for row in breakdown_rows:
        print(
            f"{row['pc_interval']:<18} {row['count']:>6d} "
            f"{format_breakdown_value(row['bias']):>9} "
            f"{format_breakdown_value(row['sigma']):>8} "
            f"{format_breakdown_value(row['within_1sigma'], '%'):>8} "
            f"{format_breakdown_value(row['within_2sigma'], '%'):>8}"
        )



def plot_eval_results(log_preds, log_gts, metrics, breakdown_rows):
    log10_errors     = np.abs(log_preds - log_gts)
    log10_signed     = log_preds - log_gts
    sigma            = metrics['sigma']
    mean_bias        = metrics['mean_bias']
    scatter_fontsize = 14
    title_fontsize   = 16
    legend_fontsize  = 11
    tick_fontsize    = 12

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 2.2])
    ax  = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Scatter: pred vs gt
    lims    = [-10, 0]
    scatter = ax.scatter(log_gts, log_preds, c=log10_errors, cmap='RdYlGn_r',
                         alpha=0.6, s=12, vmin=0, vmax=3)
    ax.plot(lims, lims,                        'r--', linewidth=1,   label='Perfect')
    ax.plot(lims, [l + sigma for l in lims],   'b:',  linewidth=0.8, alpha=0.5,
            label=f'±1σ ({sigma:.2f})')
    ax.plot(lims, [l - sigma for l in lims],   'b:',  linewidth=0.8, alpha=0.5)
    ax.plot(lims, [l + 2*sigma for l in lims], 'g:',  linewidth=0.8, alpha=0.3,
            label=f'±2σ ({2*sigma:.2f})')
    ax.plot(lims, [l - 2*sigma for l in lims], 'g:',  linewidth=0.8, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground Truth log10(Pc)', fontsize=scatter_fontsize)
    ax.set_ylabel('Predicted log10(Pc)',    fontsize=scatter_fontsize)
    ax.set_title(f'Prediction vs GT (1σ={sigma:.3f} orders)', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('log10 error', fontsize=scatter_fontsize)
    colorbar.ax.tick_params(labelsize=tick_fontsize)

    # Histogram: signed error distribution
    ax2.hist(log10_signed, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=sigma,     color='b',      linestyle='--', alpha=0.7, label=f'±1σ ({sigma:.3f})')
    ax2.axvline(x=-sigma,    color='b',      linestyle='--', alpha=0.7)
    ax2.axvline(x=2*sigma,   color='g',      linestyle='--', alpha=0.5, label=f'±2σ ({2*sigma:.3f})')
    ax2.axvline(x=-2*sigma,  color='g',      linestyle='--', alpha=0.5)
    ax2.axvline(x=mean_bias, color='orange', linestyle='-',  linewidth=2,
                label=f'Bias={mean_bias:+.3f}')
    ax2.set_xlabel('Signed Log10 Error (orders of magnitude)', fontsize=scatter_fontsize)
    ax2.set_ylabel('Count',                                    fontsize=scatter_fontsize)
    ax2.set_title('Error Distribution (Signed)',               fontsize=title_fontsize)
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
    plt.savefig('figures/prediction_scatter.png', dpi=150)
    plt.close()
    print(f"\nPlots saved to 'figures/prediction_scatter.png'")


def evaluate_best_model(model_param, val_features, val_labels, device='cpu', log_path=f'logs/eval_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'):

    eps                        = 1e-10
    model                      = load_best_model(model_param, device)
    all_preds, all_gts         = run_inference(model, val_features, val_labels, device)
    log_preds                  = np.log10(np.maximum(all_preds, eps))
    log_gts                    = np.log10(np.maximum(all_gts,   eps))
    metrics                    = compute_global_metrics(log_preds, log_gts)
    breakdown_rows             = compute_breakdown(log_preds, log_gts, metrics['sigma'])

    class _Tee:
        def __init__(self, stream, fh):
            self._stream = stream
            self._fh     = fh
        def write(self, data):
            self._stream.write(data)
            self._fh.write(data)
        def flush(self):
            self._stream.flush()
            self._fh.flush()

    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write(f"Model parameters: {model_param}\n")
        original_stdout = sys.stdout
        sys.stdout      = _Tee(original_stdout, fh)
        try:
            print_global_summary(metrics, len(all_gts))
            print_breakdown(breakdown_rows)
        finally:
            sys.stdout = original_stdout

    print(f"Evaluation log saved to '{log_path}'")
    plot_eval_results(log_preds, log_gts, metrics, breakdown_rows)
