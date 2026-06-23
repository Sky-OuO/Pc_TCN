import torch
import torch.nn as nn



class AsymmetricBerhuLoss(nn.Module):
    def __init__(self, delta=1.0, high_pc_threshold=-3.0, alpha_high=2.5,
                 lambda_mse=0.15, mid_range_weight=2.0,
                 mid_low=-5.5, mid_high=-3.5, lambda_high_bias=0.5,
                 high_bias_min_count=4, lambda_pmax=0.0, pmax_margin=0.75,
                 lambda_pmax_recall=0.0, pmax_recall_threshold=-2.0,
                 pmax_recall_temperature=0.5, pmax_recall_tolerance=0.25,
                 lambda_mid_over=0.0, mid_over_low=-6.0, mid_over_high=-4.0,
                 mid_over_target_margin=1.0, mid_over_pmax_margin=0.75):
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
        self.lambda_pmax         = lambda_pmax
        self.pmax_margin         = pmax_margin
        self.lambda_pmax_recall  = lambda_pmax_recall
        self.pmax_recall_threshold = pmax_recall_threshold
        self.pmax_recall_temperature = pmax_recall_temperature
        self.pmax_recall_tolerance = pmax_recall_tolerance
        self.lambda_mid_over     = lambda_mid_over
        self.mid_over_low        = mid_over_low
        self.mid_over_high       = mid_over_high
        self.mid_over_target_margin = mid_over_target_margin
        self.mid_over_pmax_margin = mid_over_pmax_margin

    def forward(self, log_predictions, log_targets, reduction='mean', log_pmax_proxy=None):
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

        if self.lambda_pmax > 0.0 and log_pmax_proxy is not None:
            pmax = log_pmax_proxy.view_as(log_predictions).squeeze(-1)
            over_pmax = torch.clamp(log_predictions.squeeze(-1) - pmax - self.pmax_margin, min=0.0)
            loss_flat = loss_flat + self.lambda_pmax * over_pmax ** 2

        if log_pmax_proxy is not None:
            preds = log_predictions.squeeze(-1)
            targets = log_targets.squeeze(-1)
            pmax = log_pmax_proxy.view_as(log_predictions).squeeze(-1)

            if self.lambda_pmax_recall > 0.0:
                temp = max(self.pmax_recall_temperature, 1e-6)
                feasible_gate = torch.sigmoid((pmax - self.pmax_recall_threshold) / temp)
                recall_floor = torch.minimum(targets, pmax + self.pmax_margin) - self.pmax_recall_tolerance
                high_recall_mask = (targets > self.pmax_recall_threshold).float()
                under_recall = torch.clamp(recall_floor - preds, min=0.0)
                loss_flat = loss_flat + self.lambda_pmax_recall * high_recall_mask * feasible_gate * under_recall ** 2

            if self.lambda_mid_over > 0.0:
                mid_over_mask = ((targets >= self.mid_over_low) & (targets <= self.mid_over_high)).float()
                target_ceiling = targets + self.mid_over_target_margin
                pmax_ceiling = pmax + self.mid_over_pmax_margin
                soft_ceiling = torch.maximum(targets, torch.minimum(target_ceiling, pmax_ceiling))
                over_mid = torch.clamp(preds - soft_ceiling, min=0.0)
                loss_flat = loss_flat + self.lambda_mid_over * mid_over_mask * over_mid ** 2

        if reduction == 'none':
            return loss_flat

        loss = loss_flat.mean()

        # 5. High-Pc bias penalty: batch-level, penalize under-prediction on high-risk events
        high_diff = diff.squeeze(-1)[high_pc_mask]
        if high_diff.numel() >= self.high_bias_min_count:
            neg_bias = torch.clamp(-high_diff.mean(), min=0.0)
            loss = loss + self.lambda_high_bias * neg_bias**2

        return loss
