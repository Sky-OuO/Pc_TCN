import torch
import torch.nn as nn


class LogSpaceHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=1.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha  

    def forward(self, log_predictions, log_targets, reduction='mean'):
        diff     = log_predictions - log_targets
        abs_diff = torch.abs(diff)
        quadratic = 0.5 * abs_diff ** 2
        power = self.delta ** (2 - self.alpha) / self.alpha * abs_diff ** self.alpha \
                + self.delta ** 2 * (1.0 / 2.0 - 1.0 / self.alpha)
        loss = torch.where(abs_diff <= self.delta, quadratic, power)
        if reduction == 'none':
            return loss.squeeze(-1)
        return loss.mean()


class AsymmetricBerhuLoss(nn.Module):
    def __init__(self, delta=1.0, high_pc_threshold=-2.0, alpha_high=2.5, lambda_mse=0.15):
        super().__init__()
        self.delta             = delta
        self.high_pc_threshold = high_pc_threshold
        self.alpha_high        = alpha_high
        self.lambda_mse        = lambda_mse

    def forward(self, log_predictions, log_targets, reduction='mean'):
        diff     = log_predictions - log_targets
        abs_diff = torch.abs(diff)
        linear    = abs_diff
        quadratic = (diff ** 2 + self.delta ** 2) / (2.0 * self.delta)
        loss = torch.where(abs_diff <= self.delta, linear, quadratic)  # (batch, 1)

        # Asymmetric upweight: under-prediction of high-Pc samples
        high_pc_mask    = (log_targets > self.high_pc_threshold).squeeze(-1) 
        under_pred_mask = (diff < 0).squeeze(-1)                               
        asym_mask       = high_pc_mask & under_pred_mask
        loss_flat       = loss.squeeze(-1)
        loss_flat       = torch.where(asym_mask, loss_flat * self.alpha_high, loss_flat)

        # Symmetric MSE anchor — penalizes bias toward zero-mean error
        mse = diff.squeeze(-1) ** 2
        loss_flat = loss_flat + self.lambda_mse * mse

        if reduction == 'none':
            return loss_flat
        return loss_flat.mean()
