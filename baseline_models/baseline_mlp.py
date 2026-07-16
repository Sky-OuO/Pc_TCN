"""
MLP baseline for Pc prediction.
Does NOT modify any existing code.
"""
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from logger import logger
from baseline_models.baseline_dataloader import load_baseline_data


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        prev = input_dim
        for i, dim in enumerate(hidden_dims):
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(0.25 if i == 0 else 0.15),
            ]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(cfg_path='config.json'):
    cfg = json.load(open(cfg_path))
    data_cfg = cfg['data']

    logger.info("=== MLP Baseline ===")

    X_train, X_val, _, _, y_train, y_val = load_baseline_data(
        data_cfg['feature_path'],
        data_cfg['label_path'],
        test_size=data_cfg['test_size'],
        random_state=data_cfg['random_state'],
        use_last_timestep=True,
        return_train=True,
    )
    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    model = MLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_mae = float('inf')
    best_state = None
    patience = 30
    patience_counter = 0
    num_epochs = 300
    batch_size = 128

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_train_t[idx])
            loss = criterion(pred, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae = float(torch.mean(torch.abs(val_pred - y_val_t)))

        scheduler.step(val_loss)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}: val_loss={val_loss:.6f}, val_mae={val_mae:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_val = model(X_val_t).cpu().numpy().flatten()

    signed = preds_val - y_val
    sigma = float(np.std(signed))
    log10_mae = float(np.mean(np.abs(signed)))
    mean_bias = float(np.mean(signed))
    within_1sigma = float(np.mean(np.abs(signed) < sigma) * 100.0)
    within_2sigma = float(np.mean(np.abs(signed) < 2 * sigma) * 100.0)

    logger.info(f"  Mean bias (μ)  : {mean_bias:+.4f} orders")
    logger.info(f"  1σ             : {sigma:.4f} orders")
    logger.info(f"  Log10-MAE      : {log10_mae:.4f} orders")
    logger.info(f"  Within 1σ      : {within_1sigma:.1f}%")
    logger.info(f"  Within 2σ      : {within_2sigma:.1f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(best_state, f'params/mlp_baseline_{timestamp}.pth')
    logger.info(f"Model saved to params/mlp_baseline_{timestamp}.pth")

    return {
        'log10_mae': log10_mae,
        'sigma': sigma,
        'mean_bias': mean_bias,
        'within_1sigma': within_1sigma,
        'within_2sigma': within_2sigma,
        'preds': preds_val,
        'gts': y_val,
    }


if __name__ == "__main__":
    from logger import setup_file_handler
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_handler(f"mlp_{ts}")
    train_mlp()
