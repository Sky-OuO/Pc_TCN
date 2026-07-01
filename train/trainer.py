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
        batch_labels   = batch_labels.to(device)
        batch_weights  = batch_weights.to(device)
        log_targets    = batch_labels

        optimizer.zero_grad()
        mixture, pred_low, pred_high, gate_logits = model(batch_features)

        # each sample trains ONLY its assigned expert
        with torch.no_grad():
            w_oracle  = torch.sigmoid((log_targets - moe_threshold) / moe_tau)
            hard_low  = (w_oracle < 0.5).squeeze(-1)
            hard_high = ~hard_low

        # Single mean over the full batch — avoids per-subset gradient imbalance
        all_preds = torch.where(hard_low.unsqueeze(-1), pred_low, pred_high)
        loss_mix  = (criterion(all_preds, log_targets) * batch_weights).mean()

        # Gate loss: BCE with soft oracle targets (on raw logits)
        loss_gate = gate_criterion(gate_logits[:, 1:2], w_oracle)

        loss = loss_mix + gate_lambda * loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, moe_threshold, device):

    model.eval()
    val_loss = 0.0
    all_log_preds, all_log_targets = [], []
    gate_correct, gate_total = 0, 0
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)
            mixture, _, _, gate_logits = model(batch_features)
            loss = criterion(mixture, log_targets).mean()
            val_loss += loss.item() * batch_features.size(0)
            all_log_preds.extend(mixture.cpu().numpy().flatten())
            all_log_targets.extend(log_targets.cpu().numpy().flatten())

            gate_pred = (gate_logits[:, 1] >= 0.0).long()   
            gate_gt   = (log_targets.squeeze(-1) >= moe_threshold).long()
            gate_correct += (gate_pred == gate_gt).sum().item()
            gate_total   += log_targets.size(0)

    val_loss /= len(val_loader.dataset)
    preds = np.array(all_log_preds)
    targets = np.array(all_log_targets)

    log_mae = np.mean(
        np.abs(preds - targets)
    )
    gate_acc = gate_correct / gate_total
    # tail metrics
    tail_mask = targets >= moe_threshold
    if tail_mask.sum() > 0:
        tail_mae = np.mean(
            np.abs(
                preds[tail_mask]
                - targets[tail_mask]
            )
        )
        tail_bias = np.mean(
            preds[tail_mask]
            - targets[tail_mask]
        )
    else:
        tail_mae = np.nan
        tail_bias = np.nan

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
                optimizer, scheduler, moe_threshold=-3.5, moe_tau=0.1, gate_lambda=1.0,
                num_epochs=200, device='cuda', patience=10, timestamp=''):

    model.to(device)
    gate_criterion = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, gate_criterion, optimizer, device,
            moe_threshold=moe_threshold, moe_tau=moe_tau, gate_lambda=gate_lambda,
        )

        val_loss, log_mae, gate_acc, tail_mae, tail_bias = _validate_one_epoch(model, val_loader, val_criterion, moe_threshold, device)

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
            logger.info(f'Train Loss: {train_loss:.6f}, VAL Loss: {val_loss:.6f}, Val MAE: {log_mae:.4f} orders, Gate Acc: {gate_acc:.4f}, Tail MAE: {tail_mae:.4f}, Tail Bias: {tail_bias:.4f}')
            logger.info('-' * 50)

        if patience_counter >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        torch.save(best_state, f'params/best_model_{timestamp}.pth')
        logger.info(f"Best model saved (val_loss={best_val_loss:.6f})")

    plot_loss_curve(train_losses, val_losses, filename=f'figures/loss_curve_{timestamp}.png')
    return model
