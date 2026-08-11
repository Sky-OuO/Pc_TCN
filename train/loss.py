import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    def __init__(self, high_pc_threshold=-3.0, alpha_high=3.0, transition_width=0.5):
        super().__init__()
        self.high_pc_threshold = high_pc_threshold
        self.alpha_high = alpha_high
        self.transition_width = transition_width

    def forward(self, log_predictions, log_targets):
        diff = log_predictions - log_targets
        abs_diff = torch.abs(diff)

        # 1. Base MSE: quadratic for all errors
        mse_loss = diff ** 2
        loss_flat = mse_loss.squeeze(-1)

        # 2. Asymmetric: penalize under-prediction of high-Pc (smoothly tapered near the threshold)
        width = max(self.transition_width, 1e-6)
        high_pc_weight = torch.sigmoid((log_targets - self.high_pc_threshold) / width).squeeze(-1)
        under_pred_mask = (diff < 0).squeeze(-1).float()

        asym_factor = 1.0 + self.alpha_high * high_pc_weight * under_pred_mask * torch.tanh(abs_diff.squeeze(-1))
        loss_flat = loss_flat * asym_factor
        return loss_flat
