#!/usr/bin/env python3
"""Bit packing utilities for efficient storage of quantized weights"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BitPacker:
    """Utilities for packing and unpacking quantized weights"""

    @staticmethod
    def pack_1bit_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Pack 1-bit weights (±1) into efficient bit representation"""
        # Convert ±1 to 0/1
        binary_weights = (weights > 0).to(torch.uint8)

        # Pack 8 bits per byte
        original_shape = binary_weights.shape
        flat_weights = binary_weights.flatten()

        # Pad to multiple of 8
        pad_size = (8 - (flat_weights.numel() % 8)) % 8
        if pad_size > 0:
            flat_weights = torch.cat([flat_weights, torch.zeros(pad_size, dtype=torch.uint8)])

        # Pack bits
        packed = torch.zeros(flat_weights.numel() // 8, dtype=torch.uint8)
        for i in range(8):
            packed |= (flat_weights[i::8] << i)

        return packed, original_shape

    @staticmethod
    def unpack_1bit_weights(packed: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """Unpack 1-bit weights from bit representation to ±1"""
        # Unpack bits
        flat_size = np.prod(original_shape)
        unpacked = torch.zeros(flat_size, dtype=torch.uint8)

        for i in range(8):
            unpacked[i::8] = (packed >> i) & 1

        # Reshape and convert to ±1
        unpacked = unpacked[:flat_size].reshape(original_shape)
        return unpacked.to(torch.float32) * 2 - 1  # Convert 0/1 to -1/+1

    @staticmethod
    def pack_4bit_weights(weights: torch.Tensor, scales: torch.Tensor,
                          zeros: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Pack 4-bit quantized weights with scales and zero points"""
        # Quantize to 4-bit integers (0-15)
        quantized = torch.clamp(
            torch.round((weights - zeros.unsqueeze(1)) / scales.unsqueeze(1)),
            0, 15
        ).to(torch.uint8)

        original_shape = quantized.shape
        flat_quantized = quantized.flatten()

        # Pack two 4-bit values per byte
        pad_size = flat_quantized.numel() % 2
        if pad_size:
            flat_quantized = torch.cat([flat_quantized, torch.zeros(1, dtype=torch.uint8)])

        packed = torch.zeros(flat_quantized.numel() // 2, dtype=torch.uint8)
        packed = flat_quantized[::2] | (flat_quantized[1::2] << 4)

        return packed, scales, zeros, original_shape

    @staticmethod
    def unpack_4bit_weights(packed: torch.Tensor, scales: torch.Tensor,
                            zeros: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
        """Unpack 4-bit weights and dequantize"""
        # Unpack 4-bit values
        flat_size = np.prod(original_shape)
        unpacked = torch.zeros(flat_size, dtype=torch.uint8)

        unpacked[::2] = packed & 0xF
        unpacked[1::2] = (packed >> 4) & 0xF

        # Reshape and dequantize
        quantized = unpacked[:flat_size].reshape(original_shape).float()
        dequantized = scales.unsqueeze(1) * quantized + zeros.unsqueeze(1)

        return dequantized


def calculate_compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """Calculate compression ratio"""
    return original_bits / compressed_bits if compressed_bits > 0 else 1.0


def estimate_memory_savings(total_params: int, tier_ratios: Tuple[float, float, float]) -> dict:
    """Estimate memory savings from 3-tier quantization with OneBit overhead"""
    fp16_ratio, int4_ratio, int1_ratio = tier_ratios

    # Original memory (FP16)
    original_memory = total_params * 16  # bits

    # Compressed memory with OneBit overhead
    # OneBit: 1-bit weights + FP16 scaling vectors (~1.0073 bits per weight)
    int1_effective_bits = 1.0073  # From OneBit paper

    compressed_memory = (
            total_params * fp16_ratio * 16 +  # FP16 tier
            total_params * int4_ratio * 4 +  # 4-bit tier
            total_params * int1_ratio * int1_effective_bits  # 1-bit tier with OneBit overhead
    )

    compression_ratio = calculate_compression_ratio(original_memory, compressed_memory)
    memory_savings = original_memory - compressed_memory

    return {
        'original_memory_mb': original_memory / 8 / 1024 / 1024,
        'compressed_memory_mb': compressed_memory / 8 / 1024 / 1024,
        'memory_savings_mb': memory_savings / 8 / 1024 / 1024,
        'compression_ratio': compression_ratio,
        'effective_bits': compressed_memory / total_params
    }
