import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train.models import GeoOnlyModel, HybridPcModel, XGBoostUncertaintyBranch
from train.dataset import HybridDataset
from train.loss import AsymmetricMSELoss
from train.trainer import train_model
from train.data_utils import load_hybrid_data
from train.evaluate import evaluate_best_model
from logger import logger, setup_file_handler
from datetime import datetime


class _GeoOnlyDataset:
    def __init__(self, geo, labels):
        self.geo = geo
        self.labels = labels

    def __len__(self):
        return len(self.geo)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.geo[idx]).float(),
                torch.tensor([self.labels[idx]], dtype=torch.float32))


def train_geo_only(model, geo_seq, labels, val_geo, val_labels, cfg, device, timestamp):
    train_ds = _GeoOnlyDataset(geo_seq, labels)
    val_ds = _GeoOnlyDataset(val_geo, val_labels)

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'],
                                   weight_decay=cfg['optimizer']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=cfg['scheduler']['patience'])

    model.to(device)
    best_mae = float('inf')
    best_state = None
    patience = 0

    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        train_loss = 0.0
        for geo, lbl in train_loader:
            geo, lbl = geo.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(geo), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * geo.size(0)
        train_loss /= len(train_ds)  

        model.eval()
        val_loss = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for geo, lbl in val_loader:
                geo, lbl = geo.to(device), lbl.to(device)
                p = model(geo)
                val_loss += torch.nn.functional.mse_loss(p, lbl, reduction='sum').item()
                preds.extend(p.cpu().numpy().flatten())
                gts.extend(lbl.cpu().numpy().flatten())
        val_loss /= len(val_ds)
        val_mae = float(np.mean(np.abs(np.array(preds) - np.array(gts))))
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"[Geo-Only] Epoch {epoch+1}: val_mae={val_mae:.4f}, best={best_mae:.4f}")

        if patience >= cfg['training']['patience']:
            logger.info(f"[Geo-Only] Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, f'params/geo_only_{timestamp}.pth')
    logger.info(f"[Geo-Only] Best model saved (Val MAE={best_mae:.4f})")


if __name__ == "__main__":
    with open('config.json') as f:
        cfg = json.load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_handler(timestamp)

    (train_geo, train_unc, train_labels,
     val_geo, val_unc, val_labels) = load_hybrid_data(
        cfg['data']['feature_path'], cfg['data']['label_path'],
        test_size=cfg['data']['test_size'], random_state=cfg['data']['random_state'])

    eps = 1e-10
    log_min, log_max = cfg['training']['log_target_min'], cfg['training']['log_target_max']
    train_labels = np.clip(np.log10(np.maximum(train_labels, eps)), log_min, log_max)
    val_labels   = np.clip(np.log10(np.maximum(val_labels, eps)), log_min, log_max)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Train geo: {train_geo.shape}, unc: {train_unc.shape}")

    model_cfg = cfg.get('model', {})
    head_cfg  = cfg.get('regression_head', {})

    # ── Phase 1: train geo-only model ──
    logger.info("\n=== Phase 1: Geo-Only Pre-training ===")
    geo_model = GeoOnlyModel(
        geo_dim=train_geo.shape[2],
        tcn_channels=model_cfg.get('num_channels', [32, 64, 128]),
        tcn_kernel=model_cfg.get('kernel_size', 3),
        tcn_dropout=model_cfg.get('dropout', 0.4),
        head_dims=head_cfg.get('dims', [64, 32]))

    train_geo_only(geo_model, train_geo, train_labels, val_geo, val_labels, cfg, device, timestamp)

    geo_model.eval()
    train_geo_preds = []
    ds_geo = _GeoOnlyDataset(train_geo, train_labels)
    loader_geo = DataLoader(ds_geo, batch_size=cfg['training']['batch_size'], shuffle=False,
                            num_workers=0, pin_memory=True)
    with torch.no_grad():
        for geo, _ in loader_geo:
            train_geo_preds.append(geo_model(geo.to(device)).cpu().numpy().flatten())
    train_geo_preds = np.concatenate(train_geo_preds)

    residuals = train_labels - train_geo_preds
    logger.info(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

    # ── Phase 2: train XGBoost on residuals ──
    logger.info("\n=== Phase 2: XGBoost on Residuals ===")
    xgb_cfg = cfg.get('xgboost', {})
    xgb_branch = XGBoostUncertaintyBranch(
        n_estimators=xgb_cfg.get('n_estimators', 300),
        max_depth=xgb_cfg.get('max_depth', 4),
        learning_rate=xgb_cfg.get('learning_rate', 0.05),
        subsample=xgb_cfg.get('subsample', 0.8),
        colsample_bytree=xgb_cfg.get('colsample_bytree', 0.8))
    xgb_branch.fit(train_unc, residuals)
    logger.info(f"XGBoost on residuals: {xgb_branch.n_trees} trees")

    train_leaves, train_xgb_pred = xgb_branch.transform(train_unc)
    val_leaves, val_xgb_pred = xgb_branch.transform(val_unc)

    max_leaves = xgb_cfg.get('max_leaves', 64)

    # ── Phase 3: train full HybridPcModel ──
    logger.info("\n=== Phase 3: HybridPcModel ===")
    train_dataset = HybridDataset(
        train_geo, train_leaves, train_xgb_pred, train_labels,
        seq_length=cfg['data']['seq_length'])
    val_dataset = HybridDataset(
        val_geo, val_leaves, val_xgb_pred, val_labels,
        seq_length=cfg['data']['seq_length'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    xgb_embed_cfg = cfg.get('xgb_leaf_embed', {})
    model = HybridPcModel(
        geo_dim=train_geo.shape[2],
        n_trees=xgb_branch.n_trees,
        max_leaves=max_leaves,
        tcn_channels=model_cfg.get('num_channels', [32, 64, 128]),
        tcn_kernel=model_cfg.get('kernel_size', 3),
        tcn_dropout=model_cfg.get('dropout', 0.4),
        leaf_embed_dim=xgb_embed_cfg.get('embed_dim', 8),
        unc_out_dim=xgb_embed_cfg.get('out_dim', 64),
        head_dims=head_cfg.get('dims', [64, 32]))
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AsymmetricMSELoss(
        high_pc_threshold=cfg['loss'].get('high_pc_threshold', -4.0),
        alpha_high=cfg['loss'].get('alpha_high', 0.0))
    val_criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'])
    scheduler = ReduceLROnPlateau(
        optimizer, mode=cfg['scheduler'].get('mode', 'min'),
        factor=cfg['scheduler'].get('factor', 0.5),
        patience=cfg['scheduler'].get('patience', 10))

    trained_model = train_model(
        model, train_loader, val_loader, criterion, val_criterion,
        optimizer, scheduler,
        num_epochs=cfg['training']['num_epochs'], device=device,
        patience=cfg['training']['patience'], timestamp=timestamp)

    evaluate_best_model(model, val_loader, val_labels, device=device, timestamp=timestamp)
