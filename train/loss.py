import torch
import torch.nn as nn



class AsymmetricBerhuLoss(nn.Module):
    def __init__(self, delta=1.0, high_pc_threshold=-3.0, alpha_high=2.5):
        super().__init__()
        self.delta = delta
        self.high_pc_threshold = high_pc_threshold
        self.alpha_high = alpha_high

    def forward(self, log_predictions, log_targets, reduction='mean'):
        diff     = log_predictions - log_targets
        abs_diff = torch.abs(diff)

        # 1. Base Berhu: linear for large errors, quadratic for small errors
        linear    = abs_diff
        quadratic = (diff ** 2 + self.delta ** 2) / (2.0 * self.delta)
        loss = torch.where(abs_diff <= self.delta, linear, quadratic)
        loss_flat = loss.squeeze(-1)

        # 2. Asymmetric: penalize under-prediction of high-Pc (tapered by magnitude)
        high_pc_mask    = (log_targets > self.high_pc_threshold).squeeze(-1)
        under_pred_mask = (diff < 0).squeeze(-1)
        asym_mask       = high_pc_mask & under_pred_mask
        
        asym_factor = torch.where(
            asym_mask,
            1.0 + self.alpha_high * torch.tanh(abs_diff.squeeze(-1)),
            torch.ones_like(loss_flat),
        )
        loss_flat = loss_flat * asym_factor

        if reduction == 'none':
            return loss_flat

        return loss_flat.mean()
