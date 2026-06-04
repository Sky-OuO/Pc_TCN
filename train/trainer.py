import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def _collect_features(model, dataset, device, batch_size=64, eps=1e-10):
    """Collect all training features and integer bin labels for FDS stats update."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_features, all_bin_labels = [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in loader:
            batch_features = batch_features.to(device)
            targets        = torch.clamp(batch_labels, min=eps)
            log_vals       = torch.log10(targets).clamp(-10.0, 0.0)
            bin_labels     = torch.clamp(((log_vals + 10.0) * 10).long(), 0, 99).squeeze(1)
            features       = model.extract_features(batch_features)
            all_features.append(features.cpu())
            all_bin_labels.append(bin_labels.cpu())
    return torch.cat(all_features), torch.cat(all_bin_labels)


def _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                     eps=1e-10, log_target_min=-9.0, log_target_max=-0.3):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels, batch_weights in train_loader:
        batch_features = batch_features.to(device)
        batch_labels   = batch_labels.to(device)
        batch_weights  = batch_weights.to(device)
        targets        = torch.clamp(batch_labels, min=eps)
        log_targets    = torch.clamp(torch.log10(targets), min=log_target_min, max=log_target_max)

        # Integer bin labels for FDS: log10(Pc) in [-10,0] -> bin in [0,99]
        bin_labels = torch.clamp(((log_targets + 10.0) * 10).long(), 0, 99).squeeze(1)

        optimizer.zero_grad()
        log_outputs     = model(batch_features, labels=bin_labels, epoch=epoch)
        loss_per_sample = criterion(log_outputs, log_targets, reduction='none')
        loss            = (loss_per_sample * batch_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)

    return train_loss / len(train_loader.dataset)


def _validate_one_epoch(model, val_loader, criterion, device, eps=1e-10,
                        log_target_min=-9.0, log_target_max=-0.3):
    model.eval()
    val_loss = 0.0
    all_log_preds, all_log_targets = [], []
    with torch.no_grad():
        for batch_features, batch_labels, _ in val_loader:
            batch_features  = batch_features.to(device)
            batch_labels    = batch_labels.to(device)
            targets_clamped = torch.clamp(batch_labels, min=eps)
            log_targets     = torch.clamp(torch.log10(targets_clamped),
                                          min=log_target_min, max=log_target_max)
            log_outputs     = model(batch_features)
            loss            = criterion(log_outputs, log_targets)
            val_loss       += loss.item() * batch_features.size(0)
            all_log_preds.extend(log_outputs.cpu().numpy().flatten())
            all_log_targets.extend(log_targets.cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)
    log_mae   = np.mean(np.abs(np.array(all_log_preds) - np.array(all_log_targets)))
    return val_loss, log_mae


def plot_loss_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses,   label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('figures/loss_curve.png', dpi=200)
    plt.close()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=50, device='cuda', patience=50,
                log_target_min=-9.0, log_target_max=-0.3):
    model.to(device)
    best_val_loss    = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                      log_target_min=log_target_min, log_target_max=log_target_max)

        if getattr(model, 'use_fds', False) and epoch >= model.FDS.start_update:
            all_feats, all_bins = _collect_features(model, train_loader.dataset, device)
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(all_feats.to(device), all_bins.to(device), epoch)
            model.train()

        val_loss, log_mae = _validate_one_epoch(model, val_loader, criterion, device,
                                                  log_target_min=log_target_min,
                                                  log_target_max=log_target_max)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        torch.save(model.state_dict(), 'params/last_model.pth')

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'params/best_model.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e}')
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Val Log10-MAE: {log_mae:.4f} orders')
            print('-' * 50)

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    plot_loss_curve(train_losses, val_losses)
    return model
