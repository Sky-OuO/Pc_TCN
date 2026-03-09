import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, log_predictions, log_targets):
        diff = log_predictions - log_targets
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    model.to(device)
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    train_losses, val_losses = [], []
    lr_history = []
    eps = 1e-10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            targets = torch.clamp(batch_labels, min=eps)
            log_targets = torch.log10(targets)

            optimizer.zero_grad()
            log_outputs = model(batch_features)           
            loss = criterion(log_outputs, log_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        torch.save(model.state_dict(), 'last_model.pth')

        model.eval()
        val_loss = 0.0
        all_log_preds, all_log_targets = [], []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                targets_clamped = torch.clamp(batch_labels, min=eps)
                log_targets = torch.log10(targets_clamped)

                log_outputs = model(batch_features)
                loss = criterion(log_outputs, log_targets)
                val_loss += loss.item() * batch_features.size(0)

                all_log_preds.extend(log_outputs.cpu().numpy().flatten())
                targets_clamped = torch.clamp(batch_labels, min=eps)
                all_log_targets.extend(torch.log10(targets_clamped).cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Compute log10-MAE as a monitoring metric
        log_mae = np.mean(np.abs(np.array(all_log_preds) - np.array(all_log_targets)))

        # Step the scheduler
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        scheduler.step()

        # early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')
    
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=200)
    plt.close()
    return model

def load_data(feature_path, label_path, test_size=0.2, random_state=42):
    features = np.load(feature_path)
    labels = np.load(label_path)
    labels = labels.astype(np.float64)

    indices = np.arange(len(features))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    # standardization
    mean = np.mean(features[train_indices], axis=(0, 1))
    std = np.std(features[train_indices], axis=(0, 1))
    std[std == 0] = 1.0
    # data normalization
    features = (features - mean) / std
    
    raw_geo = features[:, :, :-2]
    raw_age = features[:, :, -2:]

    diff_geo = np.diff(raw_geo, axis=1, prepend=raw_geo[:, :1, :])
    features = np.concatenate([raw_geo, diff_geo, raw_age], axis=2)

    train_features, val_features = features[train_indices], features[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]

    return train_features, train_labels, val_features, val_labels

def upsample_minority_class_sampler(labels):
    log_labels = np.log10(np.maximum(labels, 1e-10))
    bin_edges = [-10, -8, -7, -6, -5, -4.5, -4, -3.5, -3, -2, -1, 0]
    
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

def evaluate_best_model(model_param, val_features, val_labels, device='cpu'):
    best_model = TCN(**model_param)
    best_model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    best_model.eval()
    best_model.to(device)
    
    val_dataset = SatelliteCollisionDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_gts = []
    eps = 1e-10
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            log_prob = best_model(batch_features)
            prob = (10.0 ** log_prob).cpu().numpy().flatten()
            all_preds.extend(prob)
            all_gts.extend(batch_labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    log_preds = np.log10(np.maximum(all_preds, eps))
    log_gts = np.log10(np.maximum(all_gts, eps))
    
    # Signed errors in log10 space (pred - gt)
    log10_signed_errors = log_preds - log_gts
    log10_errors = np.abs(log10_signed_errors)
    
    # Sigma-based metrics
    mean_bias = np.mean(log10_signed_errors)
    sigma = np.std(log10_signed_errors)
    within_1sigma = np.mean(log10_errors < sigma) * 100.0
    within_2sigma = np.mean(log10_errors < 2 * sigma) * 100.0
    
    # Supplementary metrics
    log10_mae = np.mean(log10_errors)
    log10_median = np.median(log10_errors)
    
    print(f"\n{'='*60}")
    print(f"Evaluation on ALL {len(all_gts)} validation samples")
    print(f"{'='*60}")
    print(f"  Mean bias (μ):   {mean_bias:+.4f} orders of magnitude")
    print(f"  1σ:              {sigma:.4f} orders of magnitude")
    print(f"  2σ:              {2*sigma:.4f} orders of magnitude")
    print(f"  Within 1σ:       {within_1sigma:.1f}%  (ideal 68.3%)")
    print(f"  Within 2σ:       {within_2sigma:.1f}%  (ideal 95.4%)")
    print(f"  Log10-MAE:       {log10_mae:.4f} orders of magnitude")
    print(f"  Log10-Median:    {log10_median:.4f} orders of magnitude")

    
    # Per-decade breakdown
    print(f"\n{'─'*75}")
    print(f"{'Pc Range':<18} {'Count':>6} {'Bias(μ)':>9} {'1σ':>8} {'In 1σ':>8} {'In 2σ':>8}")
    print(f"{'─'*75}")
    decades = [(-10, -8),(-8, -6),(-6, -5),
               (-5, -4),(-4, -3),(-3, -2),(-2, 0)]
    for low, high in decades:
        mask = (log_gts >= low) & (log_gts < high)
        if np.sum(mask) > 0:
            d_bias = np.mean(log10_signed_errors[mask])
            d_sigma = np.std(log10_signed_errors[mask])
            d_in1s = np.mean(log10_errors[mask] < d_sigma) * 100.0
            d_in2s = np.mean(log10_errors[mask] < 2 * d_sigma) * 100.0
            print(f"[1e{low:+.0f}, 1e{high:+.0f})  {np.sum(mask):>6d}   {d_bias:>+8.4f} {d_sigma:>8.4f} {d_in1s:>7.1f}% {d_in2s:>7.1f}%")
    
    # Worst predictions analysis
    print(f"\n{'─'*70}")
    print("Top 10 WORST predictions:")
    print(f"{'─'*70}")
    worst_idx = np.argsort(log10_errors)[-10:][::-1]
    for idx in worst_idx:
        print(f"  Pred: {all_preds[idx]:.4e}  GT: {all_gts[idx]:.4e}  "
              f"log10 err: {log_preds[idx] - log_gts[idx]:+.3f}  "
              f"(GT decade: 1e{int(np.floor(log_gts[idx]))})")
    
    #Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: pred vs gt
    ax = axes[0]
    scatter = ax.scatter(log_gts, log_preds, c=log10_errors, cmap='RdYlGn_r', 
                        alpha=0.6, s=12, vmin=0, vmax=3)
    lims = [min(log_gts.min(), log_preds.min()) - 0.5, 
            max(log_gts.max(), log_preds.max()) + 0.5]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax.plot(lims, [l + sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5, label=f'±1σ ({sigma:.2f})')
    ax.plot(lims, [l - sigma for l in lims], 'b:', linewidth=0.8, alpha=0.5)
    ax.plot(lims, [l + 2*sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3, label=f'±2σ ({2*sigma:.2f})')
    ax.plot(lims, [l - 2*sigma for l in lims], 'g:', linewidth=0.8, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground Truth log10(Pc)')
    ax.set_ylabel('Predicted log10(Pc)')
    ax.set_title(f'Prediction vs GT (1σ={sigma:.3f} orders)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='log10 error')
    
    # Right: signed error distribution with sigma bands
    ax2 = axes[1]
    ax2.hist(log10_signed_errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=sigma, color='b', linestyle='--', alpha=0.7, label=f'±1σ ({sigma:.3f})')
    ax2.axvline(x=-sigma, color='b', linestyle='--', alpha=0.7)
    ax2.axvline(x=2*sigma, color='g', linestyle='--', alpha=0.5, label=f'±2σ ({2*sigma:.3f})')
    ax2.axvline(x=-2*sigma, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=mean_bias, color='orange', linestyle='-', linewidth=2, label=f'Bias={mean_bias:+.3f}')
    ax2.set_xlabel('Signed Log10 Error (orders of magnitude)')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution (Signed)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_scatter.png', dpi=150)
    plt.close()
    print(f"\nPlots saved to 'prediction_scatter.png'")

if __name__ == "__main__":
    feature_path = "features.npy"
    label_path = "labels.npy"
    train_features, train_labels, val_features, val_labels = load_data(feature_path, label_path)
    print(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation features shape: {val_features.shape}, Validation labels shape: {val_labels.shape}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_param = {
        'input_size': train_features.shape[2],
        'num_channels': [32, 32, 64, 64, 128, 128], 
        'kernel_size': 5,
        'dropout': 0.2
    }

    train_dataset = SatelliteCollisionDataset(train_features, train_labels)
    val_dataset = SatelliteCollisionDataset(val_features, val_labels)
    
    sampler = upsample_minority_class_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = TCN(**model_param)
    print(f"model parameters num: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = LogSpaceHuberLoss(delta=0.5)
    
    num_epochs = 300
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device
    )
    evaluate_best_model(model_param, val_features, val_labels, device=device)