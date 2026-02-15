#!/usr/bin/env python3
"""Loss functions for adaptive quantization based on OneBit methodology"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AdaptiveQuantizationLoss(nn.Module):
    """Combined loss for knowledge distillation in quantization"""

    def __init__(self, alpha: float = 1.0, temperature: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

        logger.debug(f"AdaptiveQuantizationLoss: alpha={alpha}, temperature={temperature}")

    def forward(self, student_outputs, teacher_outputs):
        """Compute combined knowledge distillation loss"""
        # Handle different output formats
        if hasattr(student_outputs, 'logits'):
            s_logits = student_outputs.logits
        else:
            s_logits = student_outputs

        if hasattr(teacher_outputs, 'logits'):
            t_logits = teacher_outputs.logits
        else:
            t_logits = teacher_outputs

        # Ensure same shape
        if s_logits.shape != t_logits.shape:
            logger.warning(f"Shape mismatch: student {s_logits.shape} vs teacher {t_logits.shape}")
            # Try to reshape if possible
            if s_logits.numel() == t_logits.numel():
                s_logits = s_logits.view(t_logits.shape)
            else:
                # Fallback to MSE
                return F.mse_loss(s_logits, t_logits)

        # Temperature-scaled knowledge distillation loss
        s_soft = F.log_softmax(s_logits / self.temperature, dim=-1)
        t_soft = F.softmax(t_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.temperature ** 2)

        # MSE loss for additional supervision
        mse_loss = F.mse_loss(s_logits, t_logits)

        # Combined loss (following OneBit paper)
        total_loss = kl_loss + self.alpha * mse_loss

        return total_loss


class OneBitLoss(nn.Module):
    """OneBit-specific loss function with SVID regularization"""

    def __init__(self, svid_weight: float = 0.1, temperature: float = 3.0):
        super().__init__()
        self.svid_weight = svid_weight
        self.temperature = temperature
        self.base_loss = AdaptiveQuantizationLoss(temperature=temperature)

    def forward(self, student_outputs, teacher_outputs, model=None):
        """Compute OneBit loss with SVID regularization"""
        # Standard knowledge distillation
        kd_loss = self.base_loss(student_outputs, teacher_outputs)

        total_loss = kd_loss

        # Add SVID regularization for 1-bit layers
        if model is not None and self.svid_weight > 0:
            svid_reg = self._compute_svid_regularization(model)
            total_loss = total_loss + self.svid_weight * svid_reg

        return total_loss

    def _compute_svid_regularization(self, model):
        """Compute SVID regularization term"""
        reg_loss = 0.0
        count = 0

        for module in model.modules():
            if hasattr(module, 'int1_sign') and module.int1_sign is not None:
                # Regularize scaling factors to maintain magnitude
                if hasattr(module, 'int1_input_scale'):
                    reg_loss += torch.mean(torch.abs(module.int1_input_scale))
                if hasattr(module, 'int1_output_scale'):
                    reg_loss += torch.mean(torch.abs(module.int1_output_scale))
                count += 1

        return reg_loss / count if count > 0 else torch.tensor(0.0)
