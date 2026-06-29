import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from logger import logger


def _train_one_epoch(model, train_loader, criterion, gate_criterion, optimizer, device,
                     moe_threshold=-3.5, moe_tau=0.1, gate_lambda=1.0):
    model.train()
    train_loss = 0.0

    for batch_features, batch_labels, batch_weights in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)
        log_targets = batch_labels

        optimizer.zero_grad()
        pred_low, pred_high, gate = model(batch_features)

        with torch.no_grad():
            w_oracle = torch.sigmoid((log_targets - moe_threshold) / moe_tau)
            hard_low = (w_oracle < 0.5).squeeze(-1)
            hard_high = ~hard_low

        loss_mix = torch.tensor(0.0, device=device)
        if hard_low.any():
            loss_mix += (criterion(pred_low[hard_low], log_targets[hard_low])
                                     * batch_weights[hard_low]).mean()
        if hard_high.any():
            loss_mix += (criterion(pred_high[hard_high], log_targets[hard_high])
                                      * batch_weights[hard_high]).mean()

        blend = (1.0 - gate) * pred_low + gate * pred_high
        loss_blend = (criterion(blend, log_targets) * batch_weights).mean()
        loss_gate = gate_criterion(gate, w_oracle)

        loss = loss_mix + loss_blend + gate_lambda * loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, moe_threshold, device):
    model.eval()
    val_loss = 0.0
    mixture_preds, low_preds, high_preds, gate_vals, log_targets_all = [], [], [], [], []
    gate_correct, gate_total = 0, 0

    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)
            pred_low, pred_high, gate = model(batch_features)

            mixture = (1.0 - gate) * pred_low + gate * pred_high
            loss = criterion(mixture, log_targets).mean()
            val_loss += loss.item() * batch_features.size(0)

            gate_pred = (gate >= 0.5).long().squeeze(-1)
            gate_gt = (log_targets.squeeze(-1) >= moe_threshold).long()
            gate_correct += (gate_pred == gate_gt).sum().item()
            gate_total += log_targets.size(0)

            mixture_preds.append(mixture.cpu().numpy())
            low_preds.append(pred_low.cpu().numpy())
            high_preds.append(pred_high.cpu().numpy())
            gate_vals.append(gate.cpu().numpy())
            log_targets_all.append(log_targets.cpu().numpy())

    mixture_preds = np.concatenate(mixture_preds).flatten()
    low_preds = np.concatenate(low_preds).flatten()
    high_preds = np.concatenate(high_preds).flatten()
    gate_vals = np.concatenate(gate_vals).flatten()
    log_targets = np.concatenate(log_targets_all).flatten()
    val_loss /= len(val_loader.dataset)
    gate_acc = gate_correct / gate_total

    log_mae = np.mean(np.abs(mixture_preds - log_targets))

    mask_low = log_targets < moe_threshold
    mask_high = ~mask_low

    low_exp_mae = (np.mean(np.abs(low_preds[mask_low] - log_targets[mask_low]))
                   if mask_low.sum() > 0 else np.nan)
    high_exp_mae = (np.mean(np.abs(high_preds[mask_high] - log_targets[mask_high]))
                    if mask_high.sum() > 0 else np.nan)
    tail_mae = (np.mean(np.abs(mixture_preds[mask_high] - log_targets[mask_high]))
                if mask_high.sum() > 0 else np.nan)
    tail_bias = (np.mean(mixture_preds[mask_high] - log_targets[mask_high])
                 if mask_high.sum() > 0 else np.nan)

    return val_loss, log_mae, gate_acc, tail_mae, tail_bias, low_exp_mae, high_exp_mae


def plot_loss_curve(train_losses, val_losses, filename='figures/loss_curve.png'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def train_model(model, train_loader, val_loader, criterion, val_criterion,
                optimizer, scheduler, moe_threshold=-3.5, moe_tau=0.1, gate_lambda=1.0,
                num_epochs=200, device='cuda', patience=10, timestamp=''):

    model.to(device)
    gate_criterion = torch.nn.BCELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, gate_criterion, optimizer, device,
            moe_threshold, moe_tau, gate_lambda)

        val_loss, log_mae, gate_acc, tail_mae, tail_bias, low_exp_mae, high_exp_mae = _validate_one_epoch(
            model, val_loader, val_criterion, moe_threshold, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(log_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}')
            logger.info(f'Train: {train_loss:.6f} | Val: {val_loss:.6f} MAE={log_mae:.4f} GateAcc={gate_acc:.3f}')
            logger.info(f'ExpMAE: low={low_exp_mae:.4f} high={high_exp_mae:.4f} | '
                        f'Tail: MAE={tail_mae:.4f} bias={tail_bias:+.4f}')
            logger.info('-' * 50)

        if patience_counter >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        torch.save(best_state, f'params/best_model_{timestamp}.pth')
        logger.info(f"Best model saved (val_loss={best_val_loss:.6f})")

    plot_loss_curve(train_losses, val_losses, filename=f'figures/loss_curve_{timestamp}.png')
    return model
