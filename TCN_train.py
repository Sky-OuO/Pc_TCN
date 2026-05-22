import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if residual.shape[2] != out.shape[2]:
            crop = out.shape[2] - residual.shape[2]
            out = out[:, :, :-crop] if crop > 0 else out
        
        if self.downsample:
            residual = self.downsample(residual)
            
        out += residual
        out = self.relu(out)
        return out

class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1)
        )
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)  # (batch, seq_len, channels)
        attn_weights = self.attention(x_t)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        out = (x_t * attn_weights).sum(dim=1)  # (batch, channels)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.3):
        super(TCN, self).__init__()
        
        self.age_feature_dim = 2
        self.geo_feature_dim = input_size - self.age_feature_dim
        
        layers = []
        num_levels = len(num_channels)
        layers.append(nn.Conv1d(self.geo_feature_dim, num_channels[0], 1))
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                 dilation_size, dropout))
        
        self.network = nn.Sequential(*layers)
        self.temporal_pool = TemporalAttentionPooling(num_channels[-1])
        
        # mapping age features to high dimension features
        self.age_mlp = nn.Sequential(
            nn.Linear(self.age_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16), 
            nn.ReLU()
        )
        
        # input dim = geo_feature_channels + age_feature_dim
        combined_dim = num_channels[-1] + 16
        
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        with torch.no_grad():
            self.regression_head[-1].bias.fill_(-5.0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, total_features)
        
        x_geo = x[:, :, :-self.age_feature_dim] 
        x_age = x[:, :, -self.age_feature_dim:] 
        
        # TCN process geo features
        x_geo = x_geo.transpose(1, 2)  # (batch, geo_dim, seq_len)
        out_geo = self.network(x_geo)  # (batch, channels, seq_len)
        
        # Attention pooling: aggregate information from ALL timesteps
        out_geo = self.temporal_pool(out_geo)  # (batch, channels)
        
        # MLP process age features
        x_age_flat = x_age[:, -1, :] # (batch, 2)
        out_age = self.age_mlp(x_age_flat) # (batch, 16)
        
        # feature fusion
        combined = torch.cat([out_geo, out_age], dim=1) # (batch, channels + 16)
        # final regression (outputs log10(Pc))
        out = self.regression_head(combined)  # (batch, 1) 
        
        return out

class SatelliteCollisionDataset(Dataset):
    def __init__(self, features, labels, seq_length=601):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        seq_data = self.features[idx]
        # padding or truncating to fixed length
        if len(seq_data) > self.seq_length:
            seq_data = seq_data[-self.seq_length:]
        elif len(seq_data) < self.seq_length:
            pad_length = self.seq_length - len(seq_data)
            seq_data = np.pad(seq_data, ((pad_length, 0), (0, 0)), mode='constant')

        features_tensor = torch.FloatTensor(seq_data)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        return features_tensor, label_tensor


class LogSpaceHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha  

    def forward(self, log_predictions, log_targets):
        diff = log_predictions - log_targets
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * abs_diff ** 2
        power = self.delta ** (2 - self.alpha) / self.alpha * abs_diff ** self.alpha \
                + self.delta ** 2 * (1.0 / 2.0 - 1.0 / self.alpha)
        loss = torch.where(abs_diff <= self.delta, quadratic, power)
        return loss.mean()



def _train_one_epoch(model, train_loader, criterion, optimizer, device,
                     eps=1e-10, log_target_min=-9.0, log_target_max=-0.3):
    """Run one training epoch; return the mean training loss."""
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels   = batch_labels.to(device)
        targets        = torch.clamp(batch_labels, min=eps)
        log_targets    = torch.clamp(torch.log10(targets), min=log_target_min, max=log_target_max)

        optimizer.zero_grad()
        log_outputs = model(batch_features)
        loss = criterion(log_outputs, log_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, device, eps=1e-10):
    """Run one validation epoch; return val loss and log10-MAE."""
    model.eval()
    val_loss = 0.0
    all_log_preds, all_log_targets = [], []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features  = batch_features.to(device)
            batch_labels    = batch_labels.to(device)
            targets_clamped = torch.clamp(batch_labels, min=eps)
            log_targets     = torch.log10(targets_clamped)
            log_outputs     = model(batch_features)
            loss            = criterion(log_outputs, log_targets)
            val_loss       += loss.item() * batch_features.size(0)
            all_log_preds.extend(log_outputs.cpu().numpy().flatten())
            all_log_targets.extend(torch.log10(targets_clamped).cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)
    log_mae   = np.mean(np.abs(np.array(all_log_preds) - np.array(all_log_targets)))
    return val_loss, log_mae


def plot_loss_curve(train_losses, val_losses):
    """Save the training/validation loss curve to loss_curve.png."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses,   label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=200)
    plt.close()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=50, device='cuda', patience=50,
                log_target_min=-9.0, log_target_max=-0.3):
    """Orchestrate training: loop over epochs, early stopping, and checkpointing."""
    model.to(device)
    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss        = _train_one_epoch(model, train_loader, criterion, optimizer, device,
                                            log_target_min=log_target_min, log_target_max=log_target_max)
        val_loss, log_mae = _validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        torch.save(model.state_dict(), 'last_model.pth')

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}')
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Val Log10-MAE: {log_mae:.4f} orders')
            print('-' * 50)

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    plot_loss_curve(train_losses, val_losses)
    return model

def load_raw_data(feature_path, label_path):
    """Load raw feature and label arrays from disk."""
    features = np.load(feature_path)
    labels   = np.load(label_path).astype(np.float64)
    return features, labels


def split_indices(n, test_size=0.2, random_state=42):
    """Return train/val index arrays."""
    indices = np.arange(n)
    return train_test_split(indices, test_size=test_size, random_state=random_state)


def normalize_features(features, train_indices):
    """Standardize features using statistics computed on the train split only."""
    mean = np.mean(features[train_indices], axis=(0, 1))
    std  = np.std(features[train_indices],  axis=(0, 1))
    std[std == 0] = 1.0
    return (features - mean) / std


def engineer_features(features):
    """Append first-order differences of geo features as extra channels."""
    raw_geo  = features[:, :, :-2]
    raw_age  = features[:, :, -2:]
    diff_geo = np.diff(raw_geo, axis=1, prepend=raw_geo[:, :1, :])
    return np.concatenate([raw_geo, diff_geo, raw_age], axis=2)


def load_data(feature_path, label_path, test_size=0.2, random_state=42):
    """Orchestrate loading, splitting, normalisation, and feature engineering."""
    features, labels             = load_raw_data(feature_path, label_path)
    train_indices, val_indices   = split_indices(len(features), test_size, random_state)
    features                     = normalize_features(features, train_indices)
    features                     = engineer_features(features)
    train_features, val_features = features[train_indices], features[val_indices]
    train_labels,   val_labels   = labels[train_indices],   labels[val_indices]
    return train_features, train_labels, val_features, val_labels

def upsample_minority_class_sampler(labels, bin_edges=None):
    if bin_edges is None:
        bin_edges = [-10, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2, -1, 0]
    log_labels = np.log10(np.maximum(labels, 1e-10))
    
    # Assign each sample to a bin
    bin_indices = np.digitize(log_labels, bin_edges) - 1 
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    # Count samples per bin
    bin_counts = {}
    for b in range(len(bin_edges) - 1):
        count = int(np.sum(bin_indices == b))
        if count > 0:
            bin_counts[b] = count
    
    print(f"{'─'*50}")
    print(f"Per-decade sampling distribution:")
    for b, count in sorted(bin_counts.items()):
        low, high = bin_edges[b], bin_edges[b + 1]
        print(f"  [1e{low:+.0f}, 1e{high:+.0f}): {count:>5d} samples")
    print(f"{'─'*50}")
    
    # Weight = 1 / (num_bins_with_data * count_in_bin)
    num_active_bins = len(bin_counts)
    sample_weights = np.zeros(len(labels), dtype=np.float64)
    for b, count in bin_counts.items():
        mask = bin_indices == b
        sample_weights[mask] = 1.0 / (num_active_bins * count)
    
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def format_breakdown_value(value, suffix=''):
    """Format a single cell value for the per-decade breakdown table."""
    if value is None:
        return '-'
    return f"{value:.1f}{suffix}" if suffix else f"{value:+.4f}" if value < 0 else f"{value:.4f}"


def load_best_model(model_param, device):
    """Load the best saved checkpoint and return the model in eval mode."""
    model = TCN(**model_param)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def run_inference(model, val_features, val_labels, device):
    """Run forward pass on the validation set; return (all_preds, all_gts) arrays."""
    val_dataset = SatelliteCollisionDataset(val_features, val_labels)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            log_prob = model(batch_features)
            all_preds.extend((10.0 ** log_prob).cpu().numpy().flatten())
            all_gts.extend(batch_labels.numpy().flatten())
    return np.array(all_preds), np.array(all_gts)


def compute_global_metrics(log_preds, log_gts):
    """Return a dict of global evaluation metrics in log10 space."""
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


def compute_breakdown(log_preds, log_gts):
    """Return per-decade breakdown rows as a list of dicts."""
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
                'within_1sigma': float(np.mean(errors[mask] < d_sigma) * 100.0),
                'within_2sigma': float(np.mean(errors[mask] < 2 * d_sigma) * 100.0),
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
    """Print the overall evaluation metrics to stdout."""
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
    """Print the per-decade breakdown table to stdout."""
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


def print_worst_predictions(all_preds, all_gts, log_preds, log_gts, n=10):
    """Print the n worst predictions ranked by absolute log10 error."""
    log10_errors = np.abs(log_preds - log_gts)
    worst_idx    = np.argsort(log10_errors)[-n:][::-1]
    print(f"\n{'─'*70}")
    print(f"Top {n} WORST predictions:")
    print(f"{'─'*70}")
    for idx in worst_idx:
        print(f"  Pred: {all_preds[idx]:.4e}  GT: {all_gts[idx]:.4e}  "
              f"log10 err: {log_preds[idx] - log_gts[idx]:+.3f}  "
              f"(GT decade: 1e{int(np.floor(log_gts[idx]))})")


def plot_eval_results(log_preds, log_gts, metrics, breakdown_rows):
    """Save scatter plot, error histogram and breakdown table to prediction_scatter.png."""
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
    plt.savefig('prediction_scatter.png', dpi=150)
    plt.close()
    print(f"\nPlots saved to 'prediction_scatter.png'")


def evaluate_best_model(model_param, val_features, val_labels, device='cpu'):
    """Orchestrate evaluation: load model, infer, compute metrics, print and plot."""
    eps                        = 1e-10
    model                      = load_best_model(model_param, device)
    all_preds, all_gts         = run_inference(model, val_features, val_labels, device)
    log_preds                  = np.log10(np.maximum(all_preds, eps))
    log_gts                    = np.log10(np.maximum(all_gts,   eps))
    metrics                    = compute_global_metrics(log_preds, log_gts)
    breakdown_rows             = compute_breakdown(log_preds, log_gts)

    print_global_summary(metrics, len(all_gts))
    print_breakdown(breakdown_rows)
    print_worst_predictions(all_preds, all_gts, log_preds, log_gts)
    plot_eval_results(log_preds, log_gts, metrics, breakdown_rows)

if __name__ == "__main__":
    with open('config.json') as f:
        cfg = json.load(f)

    train_features, train_labels, val_features, val_labels = load_data(
        cfg['data']['feature_path'],
        cfg['data']['label_path'],
        test_size=cfg['data']['test_size'],
        random_state=cfg['data']['random_state'],
    )
    print(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {val_features.shape}, Validation labels shape: {val_labels.shape}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_param = {
        'input_size':    train_features.shape[2],
        'num_channels':  cfg['model']['num_channels'],
        'kernel_size':   cfg['model']['kernel_size'],
        'dropout':       cfg['model']['dropout'],
    }

    train_dataset = SatelliteCollisionDataset(
        train_features, train_labels, seq_length=cfg['data']['seq_length'])
    val_dataset = SatelliteCollisionDataset(
        val_features, val_labels, seq_length=cfg['data']['seq_length'])

    sampler = upsample_minority_class_sampler(
        train_labels, bin_edges=cfg['sampler']['bin_edges'])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], sampler=sampler, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = TCN(**model_param)
    print(f"model parameters num: {sum(p.numel() for p in model.parameters()):,}")

    criterion = LogSpaceHuberLoss(
        delta=cfg['loss']['delta'], alpha=cfg['loss']['alpha'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
    )
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=cfg['scheduler']['warmup_start_factor'],
        total_iters=cfg['scheduler']['warmup_total_iters'],
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg['scheduler']['cosine_T_0'],
        T_mult=cfg['scheduler']['cosine_T_mult'],
        eta_min=cfg['scheduler']['cosine_eta_min'],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg['scheduler']['milestone']],
    )

    # trained_model = train_model(
    #     model, train_loader, val_loader, criterion, optimizer, scheduler,
    #     num_epochs=cfg['training']['num_epochs'], device=device,
    #     patience=cfg['training']['patience'],
    #     log_target_min=cfg['training']['log_target_min'],
    #     log_target_max=cfg['training']['log_target_max'],
    # )
    evaluate_best_model(model_param, val_features, val_labels, device=device)