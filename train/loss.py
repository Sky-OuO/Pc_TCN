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
