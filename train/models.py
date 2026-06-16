import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from logger import logger


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
    def __init__(self, channels, max_seq_len=601):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1)
        )
        # Learnable per-position bias: position 0 = furthest from TCA, last = TCA
        self.pos_bias = nn.Parameter(torch.zeros(max_seq_len, 1))
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)  # (batch, seq_len, channels)
        seq_len = x_t.size(1)
        attn_weights = self.attention(x_t)  # (batch, seq_len, 1)
        attn_weights = attn_weights + self.pos_bias[:seq_len]  # time-to-TCA positional bias
        attn_weights = F.softmax(attn_weights, dim=1)
        out = (x_t * attn_weights).sum(dim=1)  # (batch, channels)
        return out


class CrossAttentionUncertaintyEncoder(nn.Module):
    def __init__(self, input_dim=9, d_model=32, num_heads=4, output_dim=128, dropout=0.1, num_layers=2):
        super().__init__()
        self.pre_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Stacked cross-attention layers
        self.attn_layers_1to2 = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.attn_layers_2to1 = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms_a = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2 * num_layers)])
        self.norms_b = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2 * num_layers)])
        self.ffns_a = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            for _ in range(num_layers)
        ])
        self.ffns_b = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            for _ in range(num_layers)
        ])
        # Time-aware pooling instead of naive mean
        self.pool_a = TemporalAttentionPooling(d_model)
        self.pool_b = TemporalAttentionPooling(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, unc1, unc2):
        # unc1, unc2: (batch, seq_len, input_dim)
        ctx1 = self.pre_encoder(unc1)  # (batch, seq_len, d_model)
        ctx2 = self.pre_encoder(unc2)

        for i, (attn_1to2, attn_2to1, ffn_a, ffn_b) in enumerate(
                zip(self.attn_layers_1to2, self.attn_layers_2to1, self.ffns_a, self.ffns_b)):
            # obj1 attends to obj2
            attn_out, _ = attn_1to2(ctx1, ctx2, ctx2)
            ctx1 = self.norms_a[2 * i](ctx1 + attn_out)
            ctx1 = self.norms_a[2 * i + 1](ctx1 + ffn_a(ctx1))
            # obj2 attends to obj1
            attn_out, _ = attn_2to1(ctx2, ctx1, ctx1)
            ctx2 = self.norms_b[2 * i](ctx2 + attn_out)
            ctx2 = self.norms_b[2 * i + 1](ctx2 + ffn_b(ctx2))

        # TemporalAttentionPooling expects (batch, channels, seq_len)
        ctx1_pooled = self.pool_a(ctx1.transpose(1, 2))  # (batch, d_model)
        ctx2_pooled = self.pool_b(ctx2.transpose(1, 2))  # (batch, d_model)
        return self.fusion(torch.cat([ctx1_pooled, ctx2_pooled], dim=1))  # (batch, output_dim)


class FiLM(nn.Module):
    def __init__(self, geo_dim, unc_dim=None):
        super().__init__()
        if unc_dim is None:
            unc_dim = geo_dim
        self.gamma_layer = nn.Linear(unc_dim, geo_dim)
        self.beta_layer  = nn.Linear(unc_dim, geo_dim)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def forward(self, geo_feat, uncertainty_feat):
        gamma = 1.0 + self.gamma_layer(uncertainty_feat)  # (batch, geo_dim)
        beta  = self.beta_layer(uncertainty_feat)          # (batch, geo_dim)
        if geo_feat.dim() == 3:
            # geo_feat: (batch, geo_dim, seq_len) — broadcast over time
            gamma = gamma.unsqueeze(-1)
            beta  = beta.unsqueeze(-1)
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
        self.feature_dim    = feature_dim
        self.bucket_num     = bucket_num
        self.bucket_start   = bucket_start
        self.half_ks        = (ks - 1) // 2
        self.momentum       = momentum
        self.start_update   = start_update
        self.start_smooth   = start_smooth

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
        logger.debug(f"FDS: updated smoothed statistics at epoch {epoch}")

    def reset_running_stats(self):
        """Reset all statistics buffers for a clean Stage 2 start."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1.0)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1.0)
        self.num_samples_tracked.zero_()
        logger.debug("FDS: running stats reset for Stage 2.")

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
        logger.debug(f"FDS: updated running stats at epoch {epoch}")

    def smooth(self, features, labels, epoch):
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
                 unc_d_model=32, unc_num_heads=4, unc_dropout=0.1, unc_num_layers=2,
                 fds=False, fds_bucket_num=100, fds_ks=5, fds_sigma=2,
                 fds_momentum=0.9, fds_start_update=0, fds_start_smooth=1,
                 head_dims=None):
        super(TCN, self).__init__()

        self.unc_feature_dim = 14   # 7-dim per object x 2 objects (4 raw + 3 phase)
        self.geo_feature_dim = input_size - self.unc_feature_dim
        self.use_fds         = fds

        self.unc_mlp = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )

        num_levels = len(num_channels)
        split_idx  = num_levels // 2  # split point for multi-scale FiLM
        out_channels = num_channels[-1]
        mid_channels = num_channels[split_idx - 1]  # channel dim at the split

        # Stem + early TCN blocks
        stem = nn.Conv1d(self.geo_feature_dim, num_channels[0], 1)
        early_blocks = []
        for i in range(split_idx):
            in_ch  = num_channels[i - 1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            early_blocks.append(TCNBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))

        # Late TCN blocks
        late_blocks = []
        for i in range(split_idx, num_levels):
            in_ch  = num_channels[i - 1]
            out_ch = num_channels[i]
            late_blocks.append(TCNBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))

        self.network_early = nn.Sequential(stem, *early_blocks)
        self.network_late  = nn.Sequential(*late_blocks)
        self.temporal_pool = TemporalAttentionPooling(out_channels)

        self.uncertainty_encoder = CrossAttentionUncertaintyEncoder(
            input_dim=15,  # 8 MLP + 4 raw residual + 3 phase
            d_model=unc_d_model,
            num_heads=unc_num_heads,
            output_dim=out_channels,
            dropout=unc_dropout,
            num_layers=unc_num_layers,
        )
        # Mid-level FiLM: conditions intermediate geo features (seq) on uncertainty
        self.film_mid = FiLM(geo_dim=mid_channels, unc_dim=out_channels)
        # Final FiLM: conditions pooled geo features on uncertainty
        self.film     = FiLM(geo_dim=out_channels,  unc_dim=out_channels)

        # Regression head — LayerNorm instead of BatchNorm to avoid FDS interference
        _head_dims = head_dims if head_dims is not None else [128, 64]
        head_layers = []
        prev = out_channels
        _dropouts = [0.3] + [0.2] * (len(_head_dims) - 1)
        for dim, drop in zip(_head_dims, _dropouts):
            head_layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(drop)]
            prev = dim
        head_layers.append(nn.Linear(prev, 1))
        self.regression_head = nn.Sequential(*head_layers)

        with torch.no_grad():
            self.regression_head[-1].bias.fill_(-5.0)

        if fds:
            self.FDS = FDS(
                feature_dim=out_channels,
                bucket_num=fds_bucket_num,
                start_update=fds_start_update,
                start_smooth=fds_start_smooth,
                kernel='gaussian',
                ks=fds_ks,
                sigma=fds_sigma,
                momentum=fds_momentum,
            )

    def extract_features(self, x):
        x_geo  = x[:, :, :-self.unc_feature_dim]     # (batch, seq_len, geo_dim)
        x_unc1 = x[:, :, -self.unc_feature_dim:-7]   # (batch, seq_len, 7): raw(4)+phase(3)
        x_unc2 = x[:, :, -7:]                         # (batch, seq_len, 7)

        # Split raw uncertainty features (4-dim) from debris phase (3-dim)
        raw_unc1 = x_unc1[:, :, :4]   # (batch, seq_len, 4)
        phase1   = x_unc1[:, :, 4:]   # (batch, seq_len, 3)
        raw_unc2 = x_unc2[:, :, :4]   # (batch, seq_len, 4)
        phase2   = x_unc2[:, :, 4:]   # (batch, seq_len, 3)

        # MLP transforms raw features → 8-dim embedding
        emb1 = self.unc_mlp(raw_unc1)  # (batch, seq_len, 8)
        emb2 = self.unc_mlp(raw_unc2)  # (batch, seq_len, 8)

        # Residual: concatenate MLP embedding + raw features + debris phase → 15-dim
        unc1 = torch.cat([emb1, raw_unc1, phase1], dim=-1)  # (batch, seq_len, 15)
        unc2 = torch.cat([emb2, raw_unc2, phase2], dim=-1)  # (batch, seq_len, 15)

        # Uncertainty encoder runs on 15-dim full sequence
        fusion_unc = self.uncertainty_encoder(unc1, unc2)  # (batch, out_channels)

        # Geo early path
        x_geo     = x_geo.transpose(1, 2)                       # (batch, geo_dim, seq_len)
        out_early  = self.network_early(x_geo)                   # (batch, mid_channels, seq_len)

        # Mid-level FiLM: broadcast uncertainty over time dimension
        out_early  = self.film_mid(out_early, fusion_unc)        # (batch, mid_channels, seq_len)

        # Geo late path + temporal pooling
        out_late   = self.network_late(out_early)                 # (batch, out_channels, seq_len)
        out_geo    = self.temporal_pool(out_late)                 # (batch, out_channels)

        # Final FiLM on pooled geo features
        return self.film(out_geo, fusion_unc)                    # (batch, out_channels)

    def freeze_backbone(self):
        for module in [self.network_early, self.network_late,
                       self.temporal_pool, self.uncertainty_encoder,
                       self.unc_mlp, self.film_mid, self.film]:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Backbone frozen for Stage 2 decoupled training.")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")

    def forward(self, x, labels=None, epoch=0):
        # x: (batch, seq_len, total_features)
        out = self.extract_features(x)
        if self.use_fds and self.training and epoch >= self.FDS.start_smooth and labels is not None:
            out = self.FDS.smooth(out, labels, epoch)
        return self.regression_head(out)
