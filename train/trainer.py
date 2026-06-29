import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from logger import logger


# ═══════════════════════════════════════════════════════════════
# Stage 1: train backbone + binary risk classifier
# ═══════════════════════════════════════════════════════════════

def _train_stage1_epoch(model, train_loader, cls_criterion, optimizer, device,
                        risk_threshold=-4.0):
    model.train()
    total_loss = 0.0
    for batch_features, batch_labels, _ in train_loader:
        batch_features = batch_features.to(device)
        log_targets = batch_labels.to(device)
        risk_gt = (log_targets >= risk_threshold).float()  # (B, 1)

        optimizer.zero_grad()
        risk_logit, _, _ = model(batch_features)
        loss = cls_criterion(risk_logit, risk_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_features.size(0)
    return total_loss / len(train_loader.dataset)


def _validate_stage1(model, val_loader, cls_criterion, risk_threshold, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)
            risk_gt = (log_targets >= risk_threshold).float()
            risk_logit, _, _ = model(batch_features)
            loss = cls_criterion(risk_logit, risk_gt)
            val_loss += loss.item() * batch_features.size(0)
            risk_prob = torch.sigmoid(risk_logit)
            pred = (risk_prob >= 0.5).float()
            correct += (pred == risk_gt).sum().item()
            total += risk_gt.size(0)
    return val_loss / len(val_loader.dataset), correct / total


# ═══════════════════════════════════════════════════════════════
# Stage 2: train regression experts with oracle routing (classifier frozen)
# ═══════════════════════════════════════════════════════════════

def _train_stage2_epoch(model, train_loader, reg_criterion, optimizer, device,
                        risk_threshold=-4.0):
    model.train()
    total_loss = 0.0
    for batch_features, batch_labels, batch_weights in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)
        log_targets = batch_labels

        optimizer.zero_grad()
        _, pred_low, pred_high = model(batch_features)

        mask_low = (log_targets < risk_threshold).squeeze(-1)
        mask_high = ~mask_low

        loss = torch.tensor(0.0, device=device)
        if mask_low.any():
            loss = loss + (reg_criterion(pred_low[mask_low], log_targets[mask_low])
                           * batch_weights[mask_low]).mean()
        if mask_high.any():
            loss = loss + (reg_criterion(pred_high[mask_high], log_targets[mask_high])
                            * batch_weights[mask_high]).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_features.size(0)
    return total_loss / len(train_loader.dataset)


def _validate_regression(model, val_loader, val_criterion, risk_threshold, device):
    """Validation uses frozen classifier for routing — no GT routing info."""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_gts, all_low, all_high = [], [], [], []

    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)
            risk_logit, pred_low, pred_high = model(batch_features)

            # Classifier-based routing (apply sigmoid to logit)
            risk_prob = torch.sigmoid(risk_logit)
            use_high = (risk_prob >= 0.5).squeeze(-1)
            output = torch.where(use_high, pred_high.squeeze(-1), pred_low.squeeze(-1))

            loss = val_criterion(output.unsqueeze(-1), log_targets)
            val_loss += loss.item() * batch_features.size(0)

            risk_gt = (log_targets >= risk_threshold).float()
            pred_cls = (risk_prob >= 0.5).float()
            correct += (pred_cls == risk_gt).sum().item()
            total += risk_gt.size(0)

            all_preds.append(output.cpu().numpy())
            all_gts.append(log_targets.cpu().numpy())
            all_low.append(pred_low.cpu().numpy())
            all_high.append(pred_high.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    cls_acc = correct / total

    preds = np.concatenate(all_preds).flatten()
    gts = np.concatenate(all_gts).flatten()
    lows = np.concatenate(all_low).flatten()
    highs = np.concatenate(all_high).flatten()

    log_mae = np.mean(np.abs(preds - gts))
    mask_high = gts >= risk_threshold
    mask_low = ~mask_high

    tail_mae = np.mean(np.abs(preds[mask_high] - gts[mask_high])) if mask_high.sum() > 0 else np.nan
    tail_bias = np.mean(preds[mask_high] - gts[mask_high]) if mask_high.sum() > 0 else np.nan
    low_mae = np.mean(np.abs(lows[mask_low] - gts[mask_low])) if mask_low.sum() > 0 else np.nan
    high_mae = np.mean(np.abs(highs[mask_high] - gts[mask_high])) if mask_high.sum() > 0 else np.nan

    return val_loss, log_mae, cls_acc, tail_mae, tail_bias, low_mae, high_mae


# ═══════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════

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


def train_model(model, train_loader, val_loader, reg_criterion, val_criterion,
                optimizer, scheduler, risk_threshold=-4.0,
                stage1_epochs=30, stage2_epochs=150,
                device='cuda', patience=20, timestamp=''):

    model.to(device)

    # Compute class weight from data: pos_weight = num_neg / num_pos
    train_labels_np = train_loader.dataset.labels
    num_pos = float((train_labels_np >= risk_threshold).sum())
    num_neg = float((train_labels_np < risk_threshold).sum())
    pos_weight_val = num_neg / max(num_pos, 1.0)
    cls_criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device))
    logger.info(f"Classifier pos_weight = {pos_weight_val:.1f} "
                f"(neg={int(num_neg)}, pos={int(num_pos)})")

    # ── Stage 1: train backbone + classifier ──
    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: Training binary risk classifier")
    logger.info("=" * 60)

    best_acc = 0.0
    for epoch in range(stage1_epochs):
        train_loss = _train_stage1_epoch(model, train_loader, cls_criterion,
                                         optimizer, device, risk_threshold)
        val_loss, cls_acc = _validate_stage1(model, val_loader, cls_criterion,
                                             risk_threshold, device)
        scheduler.step(val_loss)

        if cls_acc > best_acc:
            best_acc = cls_acc

        if (epoch + 1) % 5 == 0:
            logger.info(f'[S1] Epoch {epoch+1}/{stage1_epochs} | '
                        f'Train: {train_loss:.4f} Val: {val_loss:.4f} Acc: {cls_acc:.3f}')

    logger.info(f"Stage 1 complete — best classifier accuracy: {best_acc:.3f}")

    # ── Stage 2: freeze classifier, train regression experts ──
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: Training regression experts (classifier frozen)")
    logger.info("=" * 60)

    model.freeze_classifier()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(stage2_epochs):
        train_loss = _train_stage2_epoch(model, train_loader, reg_criterion,
                                         optimizer, device, risk_threshold)

        (val_loss, log_mae, cls_acc, tail_mae, tail_bias,
         low_mae, high_mae) = _validate_regression(
            model, val_loader, val_criterion, risk_threshold, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(log_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'[S2] Epoch {epoch+1}/{stage2_epochs} lr={current_lr:.2e}')
            logger.info(f'Train: {train_loss:.6f} | Val: {val_loss:.6f} MAE={log_mae:.4f} ClsAcc={cls_acc:.3f}')
            logger.info(f'ExpMAE: low={low_mae:.4f} high={high_mae:.4f} | '
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
