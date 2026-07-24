import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding))
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.drop1(self.act(self.conv1(x)))
        out = self.drop2(self.act(self.conv2(out)))

        if out.shape[2] != residual.shape[2]:
            crop = out.shape[2] - residual.shape[2]
            out = out[:, :, :-crop]
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.act(out + residual)


class ContentAttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x_t = x.transpose(1, 2)                     # (batch, seq_len, channels)
        weights = F.softmax(self.score(x_t), dim=1)   # (batch, seq_len, 1)
        return (x_t * weights).sum(dim=1)              # (batch, channels)


class GeoTCNEncoder(nn.Module):
    def __init__(self, geo_dim, num_channels=(64, 128, 256), kernel_size=5, dropout=0.3):
        super().__init__()
        stem = nn.Conv1d(geo_dim, num_channels[0], 1)
        blocks = []
        for i, ch in enumerate(num_channels):
            in_ch = num_channels[i - 1] if i > 0 else num_channels[0]
            blocks.append(TCNBlock(in_ch, ch, kernel_size, dilation=2 ** i, dropout=dropout))
        self.net = nn.Sequential(stem, *blocks)
        self.pool = ContentAttentionPool(num_channels[-1])
        self.out_dim = num_channels[-1]

    def forward(self, x_geo):
        # x_geo: (batch, seq_len, geo_dim) -> (batch, geo_dim, seq_len)
        h = self.net(x_geo.transpose(1, 2))
        return self.pool(h)  # (batch, out_dim)

class XGBoostUncertaintyBranch:
    def __init__(self, **xgb_params):
        default_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1
        )
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.n_trees = default_params["n_estimators"]

    def fit(self, unc_features: np.ndarray, log10_pc_targets: np.ndarray):
        self.model.fit(unc_features, log10_pc_targets)
        return self

    def transform(self, unc_features: np.ndarray):
        booster = self.model.get_booster()
        dmat = xgb.DMatrix(unc_features)
        leaf_indices = booster.predict(dmat, pred_leaf=True).astype(np.int64)
        raw_pred = self.model.predict(unc_features).astype(np.float32)
        return leaf_indices, raw_pred


class XGBLeafEmbedding(nn.Module):
    def __init__(self, n_trees, max_leaves, embed_dim=8, out_dim=64):
        super().__init__()
        self.leaf_embed = nn.Embedding(n_trees * max_leaves, embed_dim)
        self.n_trees = n_trees
        self.max_leaves = max_leaves
        self.proj = nn.Sequential(
            nn.Linear(n_trees * embed_dim + 1, out_dim), 
            nn.LayerNorm(out_dim),
            nn.GELU())

    def forward(self, leaf_indices, raw_pred):
        offsets = torch.arange(self.n_trees, device=leaf_indices.device) * self.max_leaves
        flat_idx = leaf_indices + offsets.unsqueeze(0)     # (batch, n_trees)
        embedded = self.leaf_embed(flat_idx).flatten(1)      # (batch, n_trees*embed_dim)
        combined = torch.cat([embedded, raw_pred], dim=-1)
        return self.proj(combined)                             # (batch, out_dim)

class RegressionHead(nn.Module):
    def __init__(self, geo_dim, unc_dim, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], geo_dim + unc_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(dropout if i == 0 else dropout * 0.7)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, geo_feat, unc_feat):
        return self.net(torch.cat([geo_feat, unc_feat], dim=-1))


class GeoOnlyModel(nn.Module):
    """Geo-only model for residual pre-training phase."""
    def __init__(self, geo_dim, tcn_channels=(64, 128, 256), tcn_kernel=5,
                 tcn_dropout=0.3, head_dims=(128, 64)):
        super().__init__()
        self.geo_encoder = GeoTCNEncoder(geo_dim, tcn_channels, tcn_kernel, tcn_dropout)
        layers, prev = [], self.geo_encoder.out_dim
        for i, h in enumerate(head_dims):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(0.3 if i == 0 else 0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, geo_seq):
        return self.head(self.geo_encoder(geo_seq))


class HybridPcModel(nn.Module):
    def __init__(self, geo_dim, n_trees, max_leaves,
                 tcn_channels=(64, 128, 256), tcn_kernel=5, tcn_dropout=0.3,
                 leaf_embed_dim=8, unc_out_dim=64, head_dims=(128, 64)):
        super().__init__()
        self.geo_encoder = GeoTCNEncoder(geo_dim, tcn_channels, tcn_kernel, tcn_dropout)
        self.unc_encoder = XGBLeafEmbedding(n_trees, max_leaves, leaf_embed_dim, unc_out_dim)
        self.head = RegressionHead(self.geo_encoder.out_dim, unc_out_dim, head_dims)

    def forward(self, geo_seq, leaf_indices, xgb_pred):
        geo_feat = self.geo_encoder(geo_seq)
        unc_feat = self.unc_encoder(leaf_indices, xgb_pred)
        return self.head(geo_feat, unc_feat)

