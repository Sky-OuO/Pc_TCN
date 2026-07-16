import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from logger import logger


def _train_one_epoch(model, train_loader, criterion, gate_criterion, optimizer, device,
                     moe_threshold=-4.0, moe_tau=0.05, gate_lambda=1.0):
    model.train()
    train_loss = 0.0
    use_moe = getattr(model, 'use_moe', True)

    for batch_features, batch_labels, batch_weights in train_loader:
        batch_features = batch_features.to(device)
        batch_labels   = batch_labels.to(device)
        batch_weights  = batch_weights.to(device)
        log_targets    = batch_labels

        optimizer.zero_grad()
        mixture, pred_low, pred_high, gate_logits = model(batch_features)

        if use_moe:
            with torch.no_grad():
                w_oracle  = torch.sigmoid((log_targets - moe_threshold) / moe_tau)
                hard_low  = (w_oracle < 0.5).squeeze(-1)

            all_preds = torch.where(hard_low.unsqueeze(-1), pred_low, pred_high)
            loss_mix  = (criterion(all_preds, log_targets) * batch_weights).mean()
            loss_gate = gate_criterion(gate_logits[:, 1:2], w_oracle)
            loss = loss_mix + gate_lambda * loss_gate
        else:
            loss = (criterion(mixture, log_targets) * batch_weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, moe_threshold, device,
                        log_matrix=False):

    model.eval()
    val_loss = 0.0
    use_moe = getattr(model, 'use_moe', True)

    all_log_preds = []
    all_log_targets = []

    gate_correct = 0
    gate_total = 0

    low_low_err = []
    low_high_err = []
    high_low_err = []
    high_high_err = []

    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)

            mixture, pred_low, pred_high, gate_logits = model(batch_features)

            loss = criterion(mixture, log_targets).mean()
            val_loss += loss.item() * batch_features.size(0)

            all_log_preds.extend(mixture.cpu().numpy().flatten())
            all_log_targets.extend(log_targets.cpu().numpy().flatten())

            if use_moe:
                gate_pred = (gate_logits[:, 1] >= 0.0).long()
                gate_gt = (log_targets.squeeze(-1) >= moe_threshold).long()
                gate_correct += (gate_pred == gate_gt).sum().item()
                gate_total += log_targets.size(0)

            if use_moe and log_matrix:
                low_mask = (log_targets.squeeze(-1) < moe_threshold)
                high_mask = ~low_mask

                if low_mask.any():
                    low_low_err.extend(
                        torch.abs(
                            pred_low[low_mask] - log_targets[low_mask]
                        ).cpu().numpy().flatten()
                    )
                    high_low_err.extend(
                        torch.abs(
                            pred_high[low_mask] - log_targets[low_mask]
                        ).cpu().numpy().flatten()
                    )
                if high_mask.any():
                    low_high_err.extend(
                        torch.abs(
                            pred_low[high_mask] - log_targets[high_mask]
                        ).cpu().numpy().flatten()
                    )
                    high_high_err.extend(
                        torch.abs(
                            pred_high[high_mask] - log_targets[high_mask]
                        ).cpu().numpy().flatten()
                    )
    val_loss /= len(val_loader.dataset)
    preds = np.array(all_log_preds)
    targets = np.array(all_log_targets)
    log_mae = np.mean(np.abs(preds - targets))
    gate_acc = gate_correct / gate_total if gate_total > 0 else float('nan')

    tail_mask = targets >= moe_threshold
    if tail_mask.sum() > 0:
        tail_mae = np.mean(np.abs(preds[tail_mask] - targets[tail_mask]))
        tail_bias = np.mean(preds[tail_mask] - targets[tail_mask])
    else:
        tail_mae = np.nan
        tail_bias = np.nan

    if use_moe and log_matrix:
        logger.info("========== Expert Specialization Matrix ==========")
        logger.info(
            f"Low Expert  on Low  region : {np.mean(low_low_err):.4f} orders"
        )
        logger.info(
            f"High Expert on Low  region : {np.mean(high_low_err):.4f} orders"
        )
        logger.info(
            f"Low Expert  on High region : {np.mean(low_high_err):.4f} orders"
        )
        logger.info(
            f"High Expert on High region : {np.mean(high_high_err):.4f} orders"
        )
        logger.info("=" * 50)
        logger.info(
            f"Low Expert specialization  : {np.mean(low_high_err) - np.mean(low_low_err):.4f}"
        )
        logger.info(
            f"High Expert specialization : {np.mean(high_low_err) - np.mean(high_high_err):.4f}"
        )
        logger.info("=" * 50)
    return val_loss, log_mae, gate_acc, tail_mae, tail_bias


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
                optimizer, scheduler, moe_threshold=-4.0, moe_tau=0.05, gate_lambda=1.0,
                num_epochs=200, device='cuda', patience=10, timestamp=''):

    model.to(device)
    gate_criterion = torch.nn.BCEWithLogitsLoss()
    best_log_mae = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, gate_criterion, optimizer, device,
                                      moe_threshold, moe_tau, gate_lambda)

        val_loss, log_mae, gate_acc, tail_mae, tail_bias = _validate_one_epoch(model, val_loader, val_criterion, moe_threshold, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(log_mae)
        if log_mae < best_log_mae:
            best_log_mae = log_mae
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone()for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}")
            logger.info(
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val MAE: {log_mae:.4f} orders, "
                f"Gate Acc: {gate_acc:.4f}, "
                f"Tail MAE: {tail_mae:.4f}, "
                f"Tail Bias: {tail_bias:.4f}"
            )
            logger.info("=" * 50)

        if patience_counter >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    # Restore the best model before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, f"params/best_model_{timestamp}.pth")
        logger.info(f"Best model saved (Val MAE={best_log_mae:.4f})")

    # Final validation with expert specialization matrix
    logger.info("\nFinal evaluation on best model:")
    _validate_one_epoch(model, val_loader, val_criterion, moe_threshold, device,
                        log_matrix=True)

    plot_loss_curve(train_losses, val_losses, filename=f"figures/loss_curve_{timestamp}.png")
    return model
