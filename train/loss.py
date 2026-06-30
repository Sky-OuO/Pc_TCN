import torch
import torch.nn as nn



class AsymmetricMSELoss(nn.Module):
    def __init__(self, high_pc_threshold=-3.0, alpha_high=3.0):
        super().__init__()
        self.high_pc_threshold = high_pc_threshold
        self.alpha_high = alpha_high

    def forward(self, log_predictions, log_targets):
        diff = log_predictions - log_targets
        abs_diff = torch.abs(diff)

        # 1. Base MSE: quadratic for all errors
        mse_loss = diff ** 2
        loss_flat = mse_loss.squeeze(-1)

        # 2. Asymmetric: penalize under-prediction of high-Pc (tapered by magnitude)
        high_pc_mask  = (log_targets > self.high_pc_threshold).squeeze(-1)
        under_pred_mask = (diff < 0).squeeze(-1)
        asym_mask  = high_pc_mask & under_pred_mask
        
        asym_factor = torch.where(
            asym_mask,
            1.0 + self.alpha_high * torch.tanh(abs_diff.squeeze(-1)),
            torch.ones_like(loss_flat),
        )
        loss_flat = loss_flat * asym_factor
        return loss_flat
