#!/usr/bin/env python3
"""
Adaptive precision linear layer implementation
Based on OneBit paper with 3-tier quantization (FP16/4-bit/1-bit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptivePrecisionLinear(nn.Module):
    """
    Linear layer with adaptive precision based on neuron importance
    Supports FP16, 4-bit, and 1-bit quantization using OneBit methodology
    """

    def __init__(self, in_features: int, out_features: int,
                 tier_masks: Dict[str, torch.Tensor], device: str = 'cpu'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # Store tier masks
        self.register_buffer('fp16_mask', tier_masks['fp16'].to(device))
        self.register_buffer('int4_mask', tier_masks['int4'].to(device))
        self.register_buffer('int1_mask', tier_masks['int1'].to(device))

        # Initialize parameters for different tiers
        self._init_parameters()

        logger.debug(f"AdaptivePrecisionLinear: {in_features}→{out_features}")
        logger.debug(f"  FP16: {self.fp16_mask.sum().item()} neurons")
        logger.debug(f"  4-bit: {self.int4_mask.sum().item()} neurons")
        logger.debug(f"  1-bit: {self.int1_mask.sum().item()} neurons")

    def _init_parameters(self):
        """Initialize parameters for different precision tiers"""
        # FP16 weights (full precision for most important neurons)
        fp16_count = self.fp16_mask.sum().item()
        if fp16_count > 0:
            self.fp16_weight = nn.Parameter(
                torch.randn(fp16_count, self.in_features, device=self.device) * 0.02
            )
        else:
            self.register_parameter('fp16_weight', None)

        # 4-bit quantization parameters
        int4_count = self.int4_mask.sum().item()
        if int4_count > 0:
            # Quantized weights (stored as int8 but represents 4-bit values 0-15)
            self.register_buffer(
                'int4_weight',
                torch.randint(0, 16, (int4_count, self.in_features),
                              dtype=torch.uint8, device=self.device)
            )
            # Scale factors for dequantization
            self.int4_scales = nn.Parameter(
                torch.randn(int4_count, device=self.device) * 0.01
            )
            # Zero points for asymmetric quantization
            self.int4_zeros = nn.Parameter(
                torch.randn(int4_count, device=self.device) * 0.01
            )
        else:
            self.register_buffer('int4_weight', None)
            self.register_parameter('int4_scales', None)
            self.register_parameter('int4_zeros', None)

        # 1-bit quantization parameters (OneBit methodology)
        int1_count = self.int1_mask.sum().item()
        if int1_count > 0:
            # Sign matrix (±1) - core of OneBit method
            self.register_buffer(
                'int1_sign',
                torch.randint(0, 2, (int1_count, self.in_features),
                              dtype=torch.int8, device=self.device) * 2 - 1  # Convert to ±1
            )
            # Input scaling factor (vector g in OneBit paper)
            self.int1_input_scale = nn.Parameter(
                torch.ones(self.in_features, device=self.device) * 0.1
            )
            # Output scaling factor (vector h in OneBit paper)
            self.int1_output_scale = nn.Parameter(
                torch.ones(int1_count, device=self.device) * 0.1
            )
        else:
            self.register_buffer('int1_sign', None)
            self.register_parameter('int1_input_scale', None)
            self.register_parameter('int1_output_scale', None)

        # Bias (shared across all tiers)
        self.bias = nn.Parameter(torch.zeros(self.out_features, device=self.device))

    def init_from_fp16(self, weight: torch.Tensor):
        """Initialize from a full-precision weight matrix using SVID"""
        logger.debug("Initializing adaptive layer from FP16 weights using SVID")

        with torch.no_grad():
            # Initialize FP16 tier
            if self.fp16_weight is not None:
                fp16_indices = torch.where(self.fp16_mask)[0]
                self.fp16_weight.data = weight[fp16_indices].clone()

            # Initialize 4-bit tier
            if self.int4_weight is not None:
                int4_indices = torch.where(self.int4_mask)[0]
                int4_weights = weight[int4_indices]

                # Quantize to 4-bit
                for i, w in enumerate(int4_weights):
                    min_val = w.min()
                    max_val = w.max()

                    # Calculate scale and zero point
                    scale = (max_val - min_val) / 15.0  # 4-bit range: 0-15
                    zero_point = min_val

                    # Quantize
                    quantized = torch.clamp(
                        torch.round((w - zero_point) / scale), 0, 15
                    ).to(torch.uint8)

                    # Store parameters
                    self.int4_weight[i] = quantized
                    self.int4_scales.data[i] = scale
                    self.int4_zeros.data[i] = zero_point

            # Initialize 1-bit tier using SVID (Sign-Value-Independent Decomposition)
            if self.int1_sign is not None:
                int1_indices = torch.where(self.int1_mask)[0]
                int1_weights = weight[int1_indices]

                # Extract sign matrix (OneBit methodology)
                self.int1_sign.data = torch.sign(int1_weights).to(torch.int8)

                # Initialize scaling factors using matrix decomposition (SVID)
                abs_weights = torch.abs(int1_weights)

                # Use SVD for rank-1 approximation: |W| ≈ a * b^T
                try:
                    U, S, Vt = torch.linalg.svd(abs_weights, full_matrices=False)
                    # Take the first component for rank-1 approximation
                    a = U[:, 0] * math.sqrt(S[0])  # Output scale (vector h)
                    b = Vt[0, :] * math.sqrt(S[0])  # Input scale (vector g)

                    self.int1_output_scale.data = a
                    self.int1_input_scale.data = b
                except Exception as e:
                    logger.warning(f"SVD failed, using mean scaling: {e}")
                    # Fallback to simple scaling
                    self.int1_output_scale.data = abs_weights.mean(dim=1)
                    self.int1_input_scale.data = torch.ones_like(self.int1_input_scale.data)

    def dequantize_int4(self) -> torch.Tensor:
        """Dequantize 4-bit weights to FP16"""
        if self.int4_weight is None:
            return torch.empty(0, self.in_features, device=self.device)

        # Dequantize: weight = scale * (quantized - zero_point)
        dequantized = self.int4_scales.unsqueeze(1) * (
                self.int4_weight.float() - self.int4_zeros.unsqueeze(1)
        )
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision (OneBit methodology for 1-bit tier)"""
        outputs = []

        # FP16 computation
        if self.fp16_weight is not None:
            fp16_out = F.linear(x, self.fp16_weight)
            outputs.append(fp16_out)

        # 4-bit computation
        if self.int4_weight is not None:
            int4_weights_fp16 = self.dequantize_int4()
            int4_out = F.linear(x, int4_weights_fp16)
            outputs.append(int4_out)

        # 1-bit computation (OneBit method)
        # Following OneBit paper: Y = ((X ⊙ g) * W_sign^T) ⊙ h
        if self.int1_sign is not None:
            # Scale input with vector g
            scaled_input = x * self.int1_input_scale.unsqueeze(0)  # Broadcast across batch

            # Matrix multiplication with sign weights (±1)
            int1_out = F.linear(scaled_input, self.int1_sign.float())

            # Scale output with vector h
            int1_out = int1_out * self.int1_output_scale.unsqueeze(0)  # Broadcast across batch
            outputs.append(int1_out)

        # Combine outputs from all tiers
        if len(outputs) == 0:
            # Fallback - shouldn't happen
            logger.warning("No quantization tiers active, using zeros")
            combined_out = torch.zeros(x.size(0), self.out_features,
                                       device=x.device, dtype=x.dtype)
        elif len(outputs) == 1:
            combined_out = outputs[0]
        else:
            # Reconstruct full output by placing each tier's output in correct positions
            combined_out = torch.zeros(x.size(0), self.out_features,
                                       device=x.device, dtype=x.dtype)

            current_idx = 0
            if self.fp16_weight is not None:
                fp16_indices = torch.where(self.fp16_mask)[0]
                combined_out[:, fp16_indices] = outputs[current_idx]
                current_idx += 1

            if self.int4_weight is not None:
                int4_indices = torch.where(self.int4_mask)[0]
                combined_out[:, int4_indices] = outputs[current_idx]
                current_idx += 1

            if self.int1_sign is not None:
                int1_indices = torch.where(self.int1_mask)[0]
                combined_out[:, int1_indices] = outputs[current_idx]
                current_idx += 1

        # Add bias
        combined_out = combined_out + self.bias

        return combined_out

    def get_compression_stats(self) -> Dict[str, float]:
        """Calculate compression statistics"""
        total_neurons = self.out_features
        fp16_neurons = self.fp16_mask.sum().item()
        int4_neurons = self.int4_mask.sum().item()
        int1_neurons = self.int1_mask.sum().item()

        # Calculate effective bit-width (including OneBit overhead)
        # OneBit: 1-bit weights + FP16 scaling vectors
        int1_overhead = (
                                    int1_neurons * 1 + self.in_features * 16 + int1_neurons * 16) / int1_neurons if int1_neurons > 0 else 0

        effective_bits = (
                                 fp16_neurons * 16 +
                                 int4_neurons * 4 +
                                 int1_neurons * int1_overhead
                         ) / total_neurons if total_neurons > 0 else 16

        compression_ratio = 16.0 / effective_bits if effective_bits > 0 else 1.0

        return {
            'total_neurons': total_neurons,
            'fp16_neurons': fp16_neurons,
            'int4_neurons': int4_neurons,
            'int1_neurons': int1_neurons,
            'effective_bits': effective_bits,
            'compression_ratio': compression_ratio,
            'fp16_percentage': fp16_neurons / total_neurons * 100,
            'int4_percentage': int4_neurons / total_neurons * 100,
            'int1_percentage': int1_neurons / total_neurons * 100,
        }

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        stats = self.get_compression_stats()
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'effective_bits={stats["effective_bits"]:.2f}, '
                f'compression={stats["compression_ratio"]:.2f}x')
