import torch
import numpy as np
import matplotlib.pyplot as plt
from logger import logger


def _train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for geo_seq, leaf_idx, xgb_pred, batch_labels, batch_weights in train_loader:
        geo_seq    = geo_seq.to(device)
        leaf_idx   = leaf_idx.to(device)
        xgb_pred   = xgb_pred.to(device)
        log_targets = batch_labels.to(device)
        batch_weights = batch_weights.to(device)

        optimizer.zero_grad()
        pred = model(geo_seq, leaf_idx, xgb_pred)
        loss = (criterion(pred, log_targets) * batch_weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * geo_seq.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for geo_seq, leaf_idx, xgb_pred, batch_labels, _ in val_loader:
            geo_seq  = geo_seq.to(device)
            leaf_idx = leaf_idx.to(device)
            xgb_pred = xgb_pred.to(device)
            log_targets = batch_labels.to(device)

            pred = model(geo_seq, leaf_idx, xgb_pred)
            loss = criterion(pred, log_targets).mean()
            val_loss += loss.item() * geo_seq.size(0)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(log_targets.cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    log_mae = float(np.mean(np.abs(preds - targets)))
    sigma   = float(np.std(preds - targets))

    tail_mask = targets >= -4.0
    tail_mae = float(np.mean(np.abs(preds[tail_mask] - targets[tail_mask]))) if tail_mask.sum() > 0 else float('nan')
    tail_bias = float(np.mean(preds[tail_mask] - targets[tail_mask])) if tail_mask.sum() > 0 else float('nan')

    return val_loss, log_mae, sigma, tail_mae, tail_bias


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
                optimizer, scheduler, num_epochs=200, device='cuda',
                patience=20, timestamp=''):

    model.to(device)
    best_log_mae = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, log_mae, sigma, tail_mae, tail_bias = _validate_one_epoch(
            model, val_loader, val_criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(log_mae)

        if log_mae < best_log_mae:
            best_log_mae = log_mae
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}")
            logger.info(
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val MAE: {log_mae:.4f} orders, "
                f"1σ: {sigma:.4f}, "
                f"Tail MAE: {tail_mae:.4f}, "
                f"Tail Bias: {tail_bias:.4f}"
            )
            logger.info("=" * 50)

        if patience_counter >= patience:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, f"params/best_model_{timestamp}.pth")
        logger.info(f"Best model saved (Val MAE={best_log_mae:.4f})")

    logger.info("\nFinal evaluation on best model:")
    _validate_one_epoch(model, val_loader, val_criterion, device)

    plot_loss_curve(train_losses, val_losses, filename=f"figures/loss_curve_{timestamp}.png")
    return model
