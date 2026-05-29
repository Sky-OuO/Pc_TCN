import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


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


class CrossAttentionUncertaintyEncoder(nn.Module):
    def __init__(self, input_dim=9, d_model=32, num_heads=4, output_dim=128, dropout=0.1):
        super().__init__()
        self.pre_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.attn_1to2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_2to1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1_a   = nn.LayerNorm(d_model)
        self.norm1_b   = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm2_a = nn.LayerNorm(d_model)
        self.norm2_b = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, unc1, unc2):
        # unc1, unc2: (batch, input_dim)
        emb1 = self.pre_encoder(unc1).unsqueeze(1)  # (batch, 1, d_model)
        emb2 = self.pre_encoder(unc2).unsqueeze(1)  # (batch, 1, d_model)

        # obj1 attends to obj2 (Q=emb1, K=emb2, V=emb2)
        attn_out, _ = self.attn_1to2(emb1, emb2, emb2)
        ctx1 = self.norm1_a(emb1 + attn_out)
        ctx1 = self.norm2_a(ctx1 + self.ffn_a(ctx1))

        # obj2 attends to obj1 (Q=emb2, K=emb1, V=emb1)
        attn_out, _ = self.attn_2to1(emb2, emb1, emb1)
        ctx2 = self.norm1_b(emb2 + attn_out)
        ctx2 = self.norm2_b(ctx2 + self.ffn_b(ctx2))

        ctx1 = ctx1.squeeze(1)  # (batch, d_model)
        ctx2 = ctx2.squeeze(1)
        return self.fusion(torch.cat([ctx1, ctx2], dim=1))  # (batch, output_dim)


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma_layer = nn.Linear(dim, dim)
        self.beta_layer  = nn.Linear(dim, dim)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def forward(self, geo_feat, uncertainty_feat):
        gamma = 1.0 + self.gamma_layer(uncertainty_feat)
        beta  = self.beta_layer(uncertainty_feat)
        return gamma * geo_feat + beta



### LDS and FDS utilities
def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10.0):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid  = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix
    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel   = [0.] * half_ks + [1.] + [0.] * half_ks
        kw            = gaussian_filter1d(base_kernel, sigma=sigma)
        kernel_window = kw / max(kw)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace       = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kw            = list(map(laplace, np.arange(-half_ks, half_ks + 1)))
        kernel_window = np.array(kw) / max(kw)
    return np.array(kernel_window, dtype=np.float32)


class FDS(nn.Module):
    def __init__(self, feature_dim, bucket_num=100, bucket_start=0,
                 start_update=0, start_smooth=1, kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super().__init__()
        self.feature_dim  = feature_dim
        self.bucket_num   = bucket_num
        self.bucket_start = bucket_start
        self.half_ks      = (ks - 1) // 2
        self.momentum     = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        kw = torch.tensor(get_lds_kernel_window(kernel, ks, sigma), dtype=torch.float32)
        self.register_buffer('kernel_window', kw)

        n = bucket_num - bucket_start
        self.register_buffer('running_mean',             torch.zeros(n, feature_dim))
        self.register_buffer('running_var',              torch.ones(n, feature_dim))
        self.register_buffer('running_mean_last_epoch',  torch.zeros(n, feature_dim))
        self.register_buffer('running_var_last_epoch',   torch.ones(n, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(n, feature_dim))
        self.register_buffer('smoothed_var_last_epoch',  torch.ones(n, feature_dim))
        self.register_buffer('num_samples_tracked',      torch.zeros(n))

    def _smooth_1d(self, x):
        w = self.kernel_window.view(1, 1, -1)
        return F.conv1d(
            F.pad(x.unsqueeze(1).permute(2, 1, 0),
                  pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=w, padding=0
        ).permute(2, 1, 0).squeeze(1)

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch  = self.running_mean.clone()
        self.running_var_last_epoch   = self.running_var.clone()
        self.smoothed_mean_last_epoch = self._smooth_1d(self.running_mean_last_epoch)
        self.smoothed_var_last_epoch  = self._smooth_1d(self.running_var_last_epoch)

    def update_last_epoch_stats(self, epoch):
        self._update_last_epoch_stats()
        print(f"FDS: updated smoothed statistics at epoch {epoch}")

    def update_running_stats(self, features, labels, epoch):
        assert self.feature_dim == features.size(1)
        assert features.size(0) == labels.size(0)
        for label in torch.unique(labels):
            lb  = int(label.item())
            if lb > self.bucket_num - 1 or lb < self.bucket_start:
                continue
            idx = lb - self.bucket_start
            if lb == self.bucket_start:
                curr_feats = features[labels <= label]
            elif lb == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_n    = curr_feats.size(0)
            curr_mean = curr_feats.mean(0)
            curr_var  = curr_feats.var(0, unbiased=(curr_n > 1))
            self.num_samples_tracked[idx] += curr_n
            factor = self.momentum if self.momentum is not None else \
                     (1 - curr_n / float(self.num_samples_tracked[idx]))
            factor = 0.0 if epoch == self.start_update else factor
            self.running_mean[idx] = (1 - factor) * curr_mean + factor * self.running_mean[idx]
            self.running_var[idx]  = (1 - factor) * curr_var  + factor * self.running_var[idx]
        print(f"FDS: updated running stats at epoch {epoch}")

    def smooth(self, features, labels, epoch):
        """Calibrate batch features using smoothed per-bin statistics."""
        if epoch < self.start_smooth:
            return features
        for label in torch.unique(labels):
            lb  = int(label.item())
            if lb > self.bucket_num - 1 or lb < self.bucket_start:
                continue
            idx = lb - self.bucket_start
            if lb == self.bucket_start:
                mask = labels <= label
            elif lb == self.bucket_num - 1:
                mask = labels >= label
            else:
                mask = labels == label
            features[mask] = calibrate_mean_var(
                features[mask],
                self.running_mean_last_epoch[idx],
                self.running_var_last_epoch[idx],
                self.smoothed_mean_last_epoch[idx],
                self.smoothed_var_last_epoch[idx],
            )
        return features


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.3,
                 unc_d_model=32, unc_num_heads=4, unc_dropout=0.1,
                 fds=False, fds_bucket_num=100, fds_ks=5, fds_sigma=2,
                 fds_momentum=0.9, fds_start_update=0, fds_start_smooth=1):
        super(TCN, self).__init__()
        
        self.unc_feature_dim = 18   # 9-dim per object x 2 objects
        self.geo_feature_dim = input_size - self.unc_feature_dim
        self.use_fds         = fds
        
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
        
        self.uncertainty_encoder = CrossAttentionUncertaintyEncoder(
            input_dim=9,
            d_model=unc_d_model,
            num_heads=unc_num_heads,
            output_dim=num_channels[-1],
            dropout=unc_dropout,
        )
        self.film = FiLM(num_channels[-1])
        
        self.regression_head = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
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

        if fds:
            self.FDS = FDS(
                feature_dim=num_channels[-1],
                bucket_num=fds_bucket_num,
                start_update=fds_start_update,
                start_smooth=fds_start_smooth,
                kernel='gaussian',
                ks=fds_ks,
                sigma=fds_sigma,
                momentum=fds_momentum,
            )

    def extract_features(self, x):
        """Return fusion features before the regression head."""
        x_geo  = x[:, :, :-self.unc_feature_dim]
        x_unc1 = x[:, -1, -self.unc_feature_dim:-9]
        x_unc2 = x[:, -1, -9:]
        x_geo   = x_geo.transpose(1, 2)
        out_geo = self.network(x_geo)
        out_geo = self.temporal_pool(out_geo)
        fusion_unc = self.uncertainty_encoder(x_unc1, x_unc2)
        return self.film(out_geo, fusion_unc)   # (batch, channels)

    def forward(self, x, labels=None, epoch=0):
        # x: (batch, seq_len, total_features)
        out = self.extract_features(x)
        if self.use_fds and self.training and epoch >= self.FDS.start_smooth and labels is not None:
            out = self.FDS.smooth(out, labels, epoch)
        return self.regression_head(out)
