import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from logger import logger


def _collect_features(model, dataset, device, batch_size=64):
    """Collect per-sample features and bin labels — used only by Stage 2 FDS."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_features, all_bin_labels = [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in loader:
            batch_features = batch_features.to(device)
            bin_labels = torch.clamp(((batch_labels + 10.0) * 10).long(), 0, 99).squeeze(1)
            features = model.extract_features(batch_features)
            all_features.append(features.cpu())
            all_bin_labels.append(bin_labels.cpu())
    return torch.cat(all_features), torch.cat(all_bin_labels)


def _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels, batch_weights in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)
        log_targets = batch_labels

        optimizer.zero_grad()
        if epoch is not None:
            bin_labels = torch.clamp(((log_targets + 10.0) * 10).long(), 0, 99).squeeze(1)
            log_outputs = model(batch_features, labels=bin_labels, epoch=epoch)
        else:
            log_outputs = model(batch_features)
        loss_per_sample = criterion(log_outputs, log_targets)
        loss = (loss_per_sample * batch_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_log_preds, all_log_targets = [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features = batch_features.to(device)
            log_targets = batch_labels.to(device)
            log_outputs = model(batch_features)
            loss = criterion(log_outputs, log_targets)
            val_loss+= loss.item() * batch_features.size(0)
            all_log_preds.extend(log_outputs.cpu().numpy().flatten())
            all_log_targets.extend(log_targets.cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)
    log_mae = np.mean(np.abs(np.array(all_log_preds) - np.array(all_log_targets)))
    return val_loss, log_mae


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

# Stage 1: standard training of the entire model (backbone + head)
def train_stage1(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 num_epochs=200, device='cuda', patience=10, timestamp=''):

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, log_mae = _validate_one_epoch(model, val_loader, criterion, device)

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
            logger.info(f'[Stage1] Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}')
            logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                        f'Val Log10-MAE: {log_mae:.4f} orders')
            logger.info('-' * 50)

        if patience_counter >= patience:
            logger.info(f"[Stage1] Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        torch.save(best_state, f'params/best_model_{timestamp}.pth')
        logger.info(f"[Stage1] Best model saved (val_loss={best_val_loss:.6f})")

    plot_loss_curve(train_losses, val_losses, filename=f'figures/loss_curve_stage1_{timestamp}.png')
    return model

# Stage 2: FDS smoothing applied only to the regression head (backbone frozen)
def train_stage2(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                 stage2_epochs=100, stage2_patience=10,
                 fds_start_update=0, fds_start_smooth=5, timestamp=''):


    model.to(device)
    model.freeze_backbone()

    if getattr(model, 'use_fds', False):
        model.FDS.reset_running_stats()
        model.FDS.start_update = fds_start_update
        model.FDS.start_smooth = fds_start_smooth

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(stage2_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device,
                                      epoch=epoch)

        if getattr(model, 'use_fds', False) and epoch >= model.FDS.start_update:
            all_feats, all_bins = _collect_features(model, train_loader.dataset, device)
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(all_feats.to(device), all_bins.to(device), epoch)
            model.train()

        val_loss, log_mae = _validate_one_epoch(model, val_loader, criterion, device)

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
            logger.info(f'[Stage2] Epoch [{epoch+1}/{stage2_epochs}] lr={current_lr:.2e}')
            logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                        f'Val Log10-MAE: {log_mae:.4f} orders')
            logger.info('-' * 50)

        if patience_counter >= stage2_patience:
            logger.info(f"[Stage2] Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        torch.save(best_state, f'params/best_model_{timestamp}.pth')
        logger.info(f"[Stage2] Best model saved (val_loss={best_val_loss:.6f})")

    plot_loss_curve(train_losses, val_losses, filename=f'figures/loss_curve_stage2_{timestamp}.png')
    return model
