import json
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from train.models import TCN
from train.dataset import SatelliteCollisionDataset, compute_lds_weights
from train.loss import AsymmetricBerhuLoss
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

    physics_cfg = cfg.get('physics_proxy', {})
    pmax_feature_index = None
    preserve_feature_indices = None
    if physics_cfg.get('enabled', False):
        raw_feature_dim = np.load(cfg['data']['feature_path'], mmap_mode='r').shape[2]
        if raw_feature_dim >= physics_cfg.get('expected_raw_feature_dim', 30):
            pmax_feature_index = physics_cfg.get('pmax_feature_index', 27)
            preserve_feature_indices = [pmax_feature_index]
        else:
            logger.info("[Physics Proxy] Skipped Pmax penalty: regenerate features to include Pmax proxy columns.")

    train_features, train_labels, val_features, val_labels = load_data(
        cfg['data']['feature_path'],
        cfg['data']['label_path'],
        test_size=cfg['data']['test_size'],
        random_state=cfg['data']['random_state'],
        preserve_feature_indices=preserve_feature_indices,
    )
    logger.info(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    logger.info(f"Validation features shape: {val_features.shape}, Validation labels shape: {val_labels.shape}")

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
        'fds_start_update': fds_cfg.get('start_update', 0),
        'fds_start_smooth': fds_cfg.get('start_smooth', 1),
        'head_dims':        head_cfg.get('dims', None),
    }
    logger.info(f"config checkpoint: {cfg}")
    lds_cfg       = cfg.get('lds', {})
    log_target_min = cfg['training']['log_target_min']
    log_target_max = cfg['training']['log_target_max']
    train_weights = compute_lds_weights(
        train_labels,
        num_bins=lds_cfg.get('num_bins', 100),
        lds_kernel=lds_cfg.get('kernel', 'gaussian'),
        lds_ks=lds_cfg.get('ks', 5),
        lds_sigma=lds_cfg.get('sigma', 2),
        log_min=log_target_min,
        log_max=log_target_max,
        lds_power=lds_cfg.get('power', 1.0),
    )
    train_dataset = SatelliteCollisionDataset(
        train_features, train_labels, seq_length=cfg['data']['seq_length'])
    val_dataset   = SatelliteCollisionDataset(
        val_features, val_labels, seq_length=cfg['data']['seq_length'])

    # WeightedRandomSampler guarantees rare samples appear in every batch.
    sampler = WeightedRandomSampler(
        weights=train_weights.tolist(),
        num_samples=len(train_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], sampler=sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = TCN(**model_param)
    logger.info(f"model parameters num: {sum(p.numel() for p in model.parameters()):,}")

    # Stage 1: always train backbone on raw features — FDS only in Stage 2
    fds_was_enabled = model.use_fds
    model.use_fds = False
    if fds_was_enabled:
        logger.info("[Stage 1] FDS temporarily disabled — backbone trains on raw features.")

    loss_cfg  = cfg['loss']
    criterion = AsymmetricBerhuLoss(
        delta=loss_cfg.get('delta', 1.0),
        high_pc_threshold=loss_cfg.get('high_pc_threshold', -2.0),
        alpha_high=loss_cfg.get('alpha_high', 2.5),
        lambda_mse=loss_cfg.get('lambda_mse', 0.15),
        mid_range_weight=loss_cfg.get('mid_range_weight', 2.0),
        mid_low=loss_cfg.get('mid_low', -6.0),
        mid_high=loss_cfg.get('mid_high', -3.0),
        lambda_high_bias=loss_cfg.get('lambda_high_bias', 0.5),
        high_bias_min_count=loss_cfg.get('high_bias_min_count', 4),
        lambda_pmax=physics_cfg.get('lambda_pmax', 0.0) if pmax_feature_index is not None else 0.0,
        pmax_margin=physics_cfg.get('pmax_margin', 0.75),
        lambda_pmax_recall=physics_cfg.get('lambda_pmax_recall', 0.0) if pmax_feature_index is not None else 0.0,
        pmax_recall_threshold=physics_cfg.get('pmax_recall_threshold', -2.0),
        pmax_recall_temperature=physics_cfg.get('pmax_recall_temperature', 0.5),
        pmax_recall_tolerance=physics_cfg.get('pmax_recall_tolerance', 0.25),
        lambda_mid_over=physics_cfg.get('lambda_mid_over', 0.0) if pmax_feature_index is not None else 0.0,
        mid_over_low=physics_cfg.get('mid_over_low', -6.0),
        mid_over_high=physics_cfg.get('mid_over_high', -4.0),
        mid_over_target_margin=physics_cfg.get('mid_over_target_margin', 1.0),
        mid_over_pmax_margin=physics_cfg.get('mid_over_pmax_margin', 0.75),
    )

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

    trained_model = train_stage1(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=cfg['training']['num_epochs'], device=device,
        patience=cfg['training']['patience'],
        log_target_min=cfg['training']['log_target_min'],
        log_target_max=cfg['training']['log_target_max'],
        timestamp=timestamp,
        pmax_feature_index=pmax_feature_index,
    )

    dt_cfg = cfg.get('decoupled_training', {})
    if dt_cfg.get('enabled', False) and fds_was_enabled:
        logger.info("\n" + "="*60)
        logger.info("Starting Stage 2: decoupled head training with FDS")
        logger.info("="*60)
        # Load the best Stage 1 backbone weights before Stage 2
        model.load_state_dict(torch.load(f'params/best_model_{timestamp}.pth',
                                          map_location=device, weights_only=True),
                                          strict=False)
        
        train_stage2(
            model, train_loader, val_loader, criterion, device,
            stage2_epochs=dt_cfg.get('stage2_epochs', 100),
            stage2_lr=dt_cfg.get('stage2_lr', 1e-4),
            stage2_patience=dt_cfg.get('stage2_patience', 30),
            fds_start_update=dt_cfg.get('fds_start_update', 0),
            fds_start_smooth=dt_cfg.get('fds_start_smooth', 5),
            log_target_min=cfg['training']['log_target_min'],
            log_target_max=cfg['training']['log_target_max'],
            timestamp=timestamp,
            pmax_feature_index=pmax_feature_index,
        )
    elif dt_cfg.get('enabled', False):
        logger.info("\n[Stage 2] Skipped — FDS is disabled in model config.")

    evaluate_best_model(model_param, val_features, val_labels, device=device,
                        seq_length=cfg['data']['seq_length'], timestamp=timestamp)
