#!/usr/bin/env python3
"""
Configuration for 3-tier adaptive quantization based on OneBit methodology
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class QuantizationConfig:
    """Configuration for adaptive quantization pipeline"""

    # Tier distribution ratios (must sum to 1.0)
    tier1_ratio: float = 0.2  # FP16 (highest importance)
    tier2_ratio: float = 0.3  # 4-bit (medium importance)
    tier3_ratio: float = 0.5  # 1-bit (lowest importance)

    # Training parameters
    training_epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 8
    alpha: float = 1.0  # Balance factor for knowledge distillation

    # Data parameters
    calibration_samples: int = 128
    max_sequence_length: int = 512

    # Device and precision
    device: str = "cpu"
    mixed_precision: bool = False

    # Model saving
    save_path: Optional[str] = None
    save_compressed_model: bool = True

    # Quantization method selection
    importance_method: str = "combined"  # "activation", "weight", "gradient", "combined"
    matrix_decomposition: str = "nmf"  # "nmf", "svd"

    # Advanced parameters
    post_layer_norm: bool = True
    use_knowledge_distillation: bool = True
    gradient_clipping: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate tier ratios sum to 1.0
        total_ratio = self.tier1_ratio + self.tier2_ratio + self.tier3_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Tier ratios must sum to 1.0, got {total_ratio}")

        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA not available, switching to CPU")
            self.device = "cpu"

        # Validate importance method
        valid_methods = ["activation", "weight", "gradient", "combined"]
        if self.importance_method not in valid_methods:
            raise ValueError(f"importance_method must be one of {valid_methods}")

        # Validate matrix decomposition method
        valid_decomp = ["nmf", "svd"]
        if self.matrix_decomposition not in valid_decomp:
            raise ValueError(f"matrix_decomposition must be one of {valid_decomp}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'tier1_ratio': self.tier1_ratio,
            'tier2_ratio': self.tier2_ratio,
            'tier3_ratio': self.tier3_ratio,
            'training_epochs': self.training_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'calibration_samples': self.calibration_samples,
            'max_sequence_length': self.max_sequence_length,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'save_path': self.save_path,
            'save_compressed_model': self.save_compressed_model,
            'importance_method': self.importance_method,
            'matrix_decomposition': self.matrix_decomposition,
            'post_layer_norm': self.post_layer_norm,
            'use_knowledge_distillation': self.use_knowledge_distillation,
            'gradient_clipping': self.gradient_clipping
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

    def get_tier_masks_info(self) -> Dict[str, str]:
        """Get information about tier assignments"""
        return {
            'tier1_fp16': f"{self.tier1_ratio:.1%} of neurons (highest importance)",
            'tier2_4bit': f"{self.tier2_ratio:.1%} of neurons (medium importance)",
            'tier3_1bit': f"{self.tier3_ratio:.1%} of neurons (lowest importance)"
        }

    def estimate_compression_ratio(self) -> float:
        """Estimate overall compression ratio"""
        # FP16 = 16 bits, 4-bit = 4 bits, 1-bit = 1 bit
        weighted_bits = (self.tier1_ratio * 16 +
                         self.tier2_ratio * 4 +
                         self.tier3_ratio * 1)
        return 16.0 / weighted_bits
