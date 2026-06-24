import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train.models import TCN
from train.dataset import SatelliteCollisionDataset, compute_lds_weights
from train.loss import AsymmetricMSELoss
from train.trainer import train_stage1, train_stage2
from train.data_utils import load_data
from train.evaluate import evaluate_best_model
from logger import logger, setup_file_handler
from datetime import datetime

if __name__ == "__main__":
    with open('config.json') as f:
        cfg = json.load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_handler(timestamp)

    train_features, train_labels, val_features, val_labels = load_data(
        cfg['data']['feature_path'],
        cfg['data']['label_path'],
        test_size=cfg['data']['test_size'],
        random_state=cfg['data']['random_state'],
    )
    logger.info(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    logger.info(f"Validation features shape: {val_features.shape}, Validation labels shape: {val_labels.shape}")

    # All downstream code (dataset, loss, trainer, evaluate) receives log-space labels.
    log_target_min = cfg['training']['log_target_min']
    log_target_max = cfg['training']['log_target_max']
    eps = 1e-10
    train_labels = np.clip(np.log10(np.maximum(train_labels, eps)), log_target_min, log_target_max)
    val_labels   = np.clip(np.log10(np.maximum(val_labels, eps)),   log_target_min, log_target_max)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    fds_cfg = cfg.get('fds', {})
    head_cfg = cfg.get('regression_head', {})
    model_param = {
        'input_size':       train_features.shape[2],
        'num_channels':     cfg['model']['num_channels'],
        'kernel_size':      cfg['model']['kernel_size'],
        'dropout':          cfg['model']['dropout'],
        'unc_d_model':      cfg['uncertainty_encoder']['d_model'],
        'unc_num_heads':    cfg['uncertainty_encoder']['num_heads'],
        'unc_dropout':      cfg['uncertainty_encoder']['dropout'],
        'unc_num_layers':   cfg['uncertainty_encoder'].get('num_layers', 2),
        'fds':              fds_cfg.get('enabled', False),
        'fds_bucket_num':   fds_cfg.get('bucket_num', 100),
        'fds_ks':           fds_cfg.get('ks', 5),
        'fds_sigma':        fds_cfg.get('sigma', 2),
        'fds_momentum':     fds_cfg.get('momentum', 0.9),
        'head_dims':        head_cfg.get('dims', None),
    }
    logger.info(f"config checkpoint: {cfg}")
    lds_cfg = cfg.get('lds', {})
    train_weights = compute_lds_weights(
        train_labels,  
        num_bins=lds_cfg.get('num_bins', 100),
        lds_kernel=lds_cfg.get('kernel', 'gaussian'),
        lds_ks=lds_cfg.get('ks', 5),
        lds_sigma=lds_cfg.get('sigma', 2),
    )
    train_dataset = SatelliteCollisionDataset(
        train_features, train_labels, seq_length=cfg['data']['seq_length'],
        sample_weights=train_weights)
    val_dataset = SatelliteCollisionDataset(
        val_features, val_labels, seq_length=cfg['data']['seq_length'])

    # LDS-weighted MSE: tail samples get higher loss weight directly.
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = TCN(**model_param)
    logger.info(f"model parameters num: {sum(p.numel() for p in model.parameters()):,}")

    # Stage 1: always train backbone on raw features — FDS only in Stage 2
    fds_was_enabled = model.use_fds
    model.use_fds = False
    if fds_was_enabled:
        logger.info("[Stage 1] FDS temporarily disabled — backbone trains on raw features.")

    criterion = AsymmetricMSELoss(
        high_pc_threshold=cfg['loss'].get('high_pc_threshold', -3.0),
        alpha_high=cfg['loss'].get('alpha_high', 3.0),
    )

    val_criterion = torch.nn.MSELoss()  

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(optimizer, 
                mode= cfg['scheduler'].get('mode', 'min'), 
                factor=cfg['scheduler'].get('factor', 0.5), 
                patience=cfg['scheduler'].get('patience', 10))

    trained_model = train_stage1(
        model, train_loader, val_loader, criterion, val_criterion,
        optimizer, scheduler,
        num_epochs=cfg['training']['num_epochs'], device=device,
        patience=cfg['training']['patience'],
        timestamp=timestamp,
    )

    dt_cfg = cfg.get('decoupled_training', {})
    if dt_cfg.get('enabled', False) and fds_was_enabled:
        logger.info("\n" + "="*60)
        logger.info("Starting Stage 2: decoupled head training with FDS")
        logger.info("="*60)
        # Load the best Stage 1 backbone weights before Stage 2
        model.load_state_dict(torch.load(f'params/best_model_{timestamp}.pth', map_location=device, weights_only=True),strict=False)

        train_stage2(
            model, train_loader, val_loader, criterion, val_criterion,
            optimizer, scheduler, device,
            stage2_epochs=dt_cfg.get('stage2_epochs', 50),
            fds_start_update=dt_cfg.get('fds_start_update', 0),
            fds_start_smooth=dt_cfg.get('fds_start_smooth', 20),
            timestamp=timestamp,
        )
    elif dt_cfg.get('enabled', False):
        logger.info("\n[Stage 2] Skipped — FDS is disabled in model config.")

    evaluate_best_model(model_param, val_features, val_labels, device=device, seq_length=cfg['data']['seq_length'], timestamp=timestamp)
