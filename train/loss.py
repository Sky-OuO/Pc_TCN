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
    def __init__(self, delta=1.0, high_pc_threshold=-3.0, alpha_high=2.5,
                 lambda_mse=0.15, mid_range_weight=2.0,
                 mid_low=-5.5, mid_high=-3.5, lambda_high_bias=0.5,
                 high_bias_min_count=4):
        super().__init__()
        self.delta               = delta
        self.high_pc_threshold   = high_pc_threshold
        self.alpha_high          = alpha_high
        self.lambda_mse          = lambda_mse
        self.mid_range_weight    = mid_range_weight
        self.mid_low             = mid_low
        self.mid_high            = mid_high
        self.lambda_high_bias    = lambda_high_bias
        self.high_bias_min_count = high_bias_min_count

    def forward(self, log_predictions, log_targets, reduction='mean'):
        diff     = log_predictions - log_targets
        abs_diff = torch.abs(diff)

        # 1. Base Berhu: linear for small errors, quadratic for large
        linear    = abs_diff
        quadratic = (diff ** 2 + self.delta ** 2) / (2.0 * self.delta)
        loss = torch.where(abs_diff <= self.delta, linear, quadratic)  # (batch, 1)
        loss_flat = loss.squeeze(-1)

        # 2. Asymmetric: under-prediction of high-Pc — tapered by error magnitude
        high_pc_mask    = (log_targets > self.high_pc_threshold).squeeze(-1)
        under_pred_mask = (diff < 0).squeeze(-1)
        asym_mask       = high_pc_mask & under_pred_mask
        # Larger under-prediction → higher penalty (1 + alpha_high * tanh(|error|))
        asym_factor = torch.where(
            asym_mask,
            1.0 + self.alpha_high * torch.tanh(abs_diff.squeeze(-1)),
            torch.ones_like(loss_flat),
        )
        loss_flat = loss_flat * asym_factor

        # 3. Middle-range upweight: extra focus on sparse transition zone
        mid_mask = (log_targets > self.mid_low) & (log_targets < self.mid_high)
        mid_mask = mid_mask.squeeze(-1)
        loss_flat = torch.where(mid_mask, loss_flat * self.mid_range_weight, loss_flat)

        # 4. Symmetric MSE anchor: directly penalizes bias
        mse = diff.squeeze(-1) ** 2
        loss_flat = loss_flat + self.lambda_mse * mse

        if reduction == 'none':
            return loss_flat

        loss = loss_flat.mean()

        # 5. High-Pc bias penalty: batch-level, penalize under-prediction on high-risk events
        high_diff = diff.squeeze(-1)[high_pc_mask]
        if high_diff.numel() >= self.high_bias_min_count:
            neg_bias = torch.clamp(-high_diff.mean(), min=0.0)
            loss = loss + self.lambda_high_bias * neg_bias**2

        return loss
