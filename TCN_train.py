import json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from train.models import TCN
from train.dataset import SatelliteCollisionDataset, compute_lds_weights
from train.loss import LogSpaceHuberLoss
from train.trainer import train_model
from train.data_utils import load_data
from train.evaluate import evaluate_best_model

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

    fds_cfg = cfg.get('fds', {})
    model_param = {
        'input_size':       train_features.shape[2],
        'num_channels':     cfg['model']['num_channels'],
        'kernel_size':      cfg['model']['kernel_size'],
        'dropout':          cfg['model']['dropout'],
        'unc_d_model':      cfg['uncertainty_encoder']['d_model'],
        'unc_num_heads':    cfg['uncertainty_encoder']['num_heads'],
        'unc_dropout':      cfg['uncertainty_encoder']['dropout'],
        'fds':              fds_cfg.get('enabled', False),
        'fds_bucket_num':   fds_cfg.get('bucket_num', 100),
        'fds_ks':           fds_cfg.get('ks', 5),
        'fds_sigma':        fds_cfg.get('sigma', 2),
        'fds_momentum':     fds_cfg.get('momentum', 0.9),
        'fds_start_update': fds_cfg.get('start_update', 0),
        'fds_start_smooth': fds_cfg.get('start_smooth', 1),
    }

    lds_cfg       = cfg.get('lds', {})
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
    val_dataset   = SatelliteCollisionDataset(
        val_features, val_labels, seq_length=cfg['data']['seq_length'])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
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

    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=cfg['training']['num_epochs'], device=device,
        patience=cfg['training']['patience'],
        log_target_min=cfg['training']['log_target_min'],
        log_target_max=cfg['training']['log_target_max'],
    )
    evaluate_best_model(model_param, val_features, val_labels, device=device)