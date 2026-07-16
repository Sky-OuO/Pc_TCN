import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train.models import TCN
from train.dataset import SatelliteCollisionDataset, compute_lds_weights
from train.loss import AsymmetricMSELoss
from train.trainer import train_model
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

    # Transform labels to log₁₀ space once
    log_target_min = cfg['training']['log_target_min']
    log_target_max = cfg['training']['log_target_max']
    eps = 1e-10
    train_labels = np.clip(np.log10(np.maximum(train_labels, eps)), log_target_min, log_target_max)
    val_labels = np.clip(np.log10(np.maximum(val_labels, eps)), log_target_min, log_target_max)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    head_cfg = cfg.get('regression_head', {})
    moe_cfg = cfg.get('moe', {})
    model_cfg = cfg.get('model', {})
    model_param = {
        'input_size': train_features.shape[2],
        'num_channels': model_cfg['num_channels'],
        'kernel_size': model_cfg['kernel_size'],
        'dropout': model_cfg['dropout'],
        'unc_d_model': cfg['uncertainty_encoder']['d_model'],
        'unc_num_heads': cfg['uncertainty_encoder']['num_heads'],
        'unc_dropout': cfg['uncertainty_encoder']['dropout'],
        'unc_num_layers': cfg['uncertainty_encoder'].get('num_layers', 2),
        'head_dims': head_cfg.get('dims', None),
        'use_moe': model_cfg.get('use_moe', True),
        'use_uncertainty_encoder': model_cfg.get('use_uncertainty_encoder', True),
        'use_film': model_cfg.get('use_film', True),
        'pos_bias_scale': model_cfg.get('pos_bias_scale', 0.0),
    }
    logger.info(f"config checkpoint: {cfg}")

    lds_cfg = cfg.get('lds', {})
    if lds_cfg.get('enabled', True):
        train_weights = compute_lds_weights(
            train_labels,
            num_bins=lds_cfg.get('num_bins', 100),
            lds_kernel=lds_cfg.get('kernel', 'gaussian'),
            lds_ks=lds_cfg.get('ks', 5),
            lds_sigma=lds_cfg.get('sigma', 2),
        )
    else:
        train_weights = None
        logger.info("LDS is disabled — using uniform sample weights.")
    train_dataset = SatelliteCollisionDataset(
        train_features, train_labels, seq_length=cfg['data']['seq_length'],
        sample_weights=train_weights)
    val_dataset = SatelliteCollisionDataset(
        val_features, val_labels, seq_length=cfg['data']['seq_length'])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = TCN(**model_param)
    logger.info(f"model parameters num: {sum(p.numel() for p in model.parameters()):,}")

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
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler'].get('mode', 'min'),
        factor=cfg['scheduler'].get('factor', 0.5),
        patience=cfg['scheduler'].get('patience', 10),
    )

    trained_model = train_model(
        model, train_loader, val_loader, criterion, val_criterion,
        optimizer, scheduler,
        moe_threshold=moe_cfg.get('threshold', -3.5),
        moe_tau=moe_cfg.get('tau', 0.1),
        gate_lambda=moe_cfg.get('gate_lambda', 2.0),
        num_epochs=cfg['training']['num_epochs'], device=device,
        patience=cfg['training']['patience'],
        timestamp=timestamp,
    )

    evaluate_best_model(model_param, val_features, val_labels, device=device,
                        seq_length=cfg['data']['seq_length'], timestamp=timestamp,
                        moe_threshold=moe_cfg.get('threshold', -3.5))
