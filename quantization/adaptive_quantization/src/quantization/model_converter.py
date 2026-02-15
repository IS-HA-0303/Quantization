#!/usr/bin/env python3
"""Model converter for 3-tier adaptive quantization"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

from .adaptive_linear import AdaptivePrecisionLinear
from config.quantization_config import QuantizationConfig

logger = logging.getLogger(__name__)


class ModelConverter:
    """Convert models to 3-tier adaptive precision using OneBit methodology"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
    def assign_tiers(self, importance_scores: Dict[str, torch.Tensor], config: QuantizationConfig) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Assign neurons to FP16, INT4, or INT1 tiers based on importance scores.

        Args:
            importance_scores: dict mapping layer name -> tensor of importance scores (higher = more important)
            config: QuantizationConfig containing device and tier ratios

        Returns:
            tier_masks: dict mapping layer_name -> { 'fp16': mask, 'int4': mask, 'int1': mask }
        """
        logger.info("Assigning tiers based on importance scores...")

        tier_masks = {}

        fp16_ratio = getattr(config, "fp16_ratio", 0.33)
        int4_ratio = getattr(config, "int4_ratio", 0.33)
        device = config.device

        for layer_name, scores in importance_scores.items():
            scores = scores.to(device)
            n = scores.numel()

            # Sort neurons by importance (descending)
            sorted_indices = torch.argsort(scores, descending=True)

            # Compute cutoff thresholds
            fp16_cut = int(fp16_ratio * n)
            int4_cut = int((fp16_ratio + int4_ratio) * n)

            fp16_mask = torch.zeros(n, dtype=torch.bool, device=device)
            int4_mask = torch.zeros(n, dtype=torch.bool, device=device)
            int1_mask = torch.zeros(n, dtype=torch.bool, device=device)

            # Assign tiers based on sorted indices
            fp16_mask[sorted_indices[:fp16_cut]] = True
            int4_mask[sorted_indices[fp16_cut:int4_cut]] = True
            int1_mask[sorted_indices[int4_cut:]] = True

            tier_masks[layer_name] = {
                "fp16": fp16_mask,
                "int4": int4_mask,
                "int1": int1_mask
            }

            logger.debug(
                f"Layer {layer_name}: "
                f"{fp16_cut} FP16, {int4_cut - fp16_cut} INT4, {n - int4_cut} INT1"
            )

        logger.info(f"Tiers successfully assigned for {len(tier_masks)} layers.")
        return tier_masks

    def convert_model(self, model: nn.Module, tier_masks: Dict[str, Dict[str, torch.Tensor]]) -> nn.Module:
        """Convert all Linear layers to adaptive precision"""
        logger.info("Converting model to 3-tier adaptive precision...")

        converted_layers = 0
        conversion_log = []

        # Get all linear layers first
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module

        logger.info(f"Found {len(linear_layers)} Linear layers to convert")

        for name, module in list(linear_layers.items()):
            if name in tier_masks:
                try:
                    logger.debug(f"Converting layer: {name}")

                    # Create adaptive layer
                    adaptive_layer = AdaptivePrecisionLinear(
                        module.in_features,
                        module.out_features,
                        tier_masks[name],
                        self.config.device
                    )

                    # Initialize from original weights using SVID
                    with torch.no_grad():
                        adaptive_layer.init_from_fp16(module.weight.detach().clone())
                        if module.bias is not None:
                            adaptive_layer.bias.data = module.bias.detach().clone()

                    # Replace in model
                    self._replace_module(model, name, adaptive_layer)

                    converted_layers += 1

                    # Log conversion details
                    stats = adaptive_layer.get_compression_stats()
                    conversion_log.append({
                        'layer': name,
                        'original_params': module.weight.numel(),
                        'effective_bits': stats['effective_bits'],
                        'compression_ratio': stats['compression_ratio']
                    })

                    logger.debug(f"✅ Converted {name}: {stats['effective_bits']:.2f} bits, "
                                 f"{stats['compression_ratio']:.2f}x compression")

                except Exception as e:
                    logger.error(f"❌ Failed to convert layer {name}: {e}")
                    continue
            else:
                logger.warning(f"⚠️  No tier mask found for layer {name}, skipping")

        logger.info(f"Successfully converted {converted_layers}/{len(linear_layers)} layers")

        # Print conversion summary
        if conversion_log:
            total_params = sum(log['original_params'] for log in conversion_log)
            avg_bits = sum(log['effective_bits'] * log['original_params'] for log in conversion_log) / total_params
            avg_compression = 16 / avg_bits if avg_bits > 0 else 1.0

            logger.info(f"📊 Conversion Summary:")
            logger.info(f"  • Converted layers: {converted_layers}")
            logger.info(f"  • Total parameters: {total_params:,}")
            logger.info(f"  • Average effective bits: {avg_bits:.2f}")
            logger.info(f"  • Average compression: {avg_compression:.2f}x")

        return model

    def calculate_compression_stats(self, tier_masks: dict) -> dict:
        """Calculate compression statistics from tier masks"""
        if not tier_masks:
            return {
                'total_neurons': 0,
                'fp16_neurons': 0,
                'int4_neurons': 0,
                'int1_neurons': 0,
                'effective_bits': 16.0,
                'compression_ratio': 1.0
            }

        total_neurons = 0
        fp16_count = 0
        int4_count = 0
        int1_count = 0

        for layer_name, masks in tier_masks.items():
            layer_total = len(masks.get('fp16', []))
            total_neurons += layer_total

            fp16_count += masks.get('fp16', torch.zeros(layer_total)).sum().item()
            int4_count += masks.get('int4', torch.zeros(layer_total)).sum().item()
            int1_count += masks.get('int1', torch.zeros(layer_total)).sum().item()

        # Calculate effective bits
        if total_neurons > 0:
            effective_bits = (fp16_count * 16 + int4_count * 4 + int1_count * 1) / total_neurons
            compression_ratio = 16.0 / effective_bits if effective_bits > 0 else 1.0
        else:
            effective_bits = 16.0
            compression_ratio = 1.0

        return {
            'total_neurons': int(total_neurons),
            'fp16_neurons': int(fp16_count),
            'int4_neurons': int(int4_count),
            'int1_neurons': int(int1_count),
            'effective_bits': float(effective_bits),
            'compression_ratio': float(compression_ratio)
        }

    def _replace_module(self, model: nn.Module, module_path: str, new_module: nn.Module):
        """Replace a module in the model by its path"""
        path_parts = module_path.split('.')
        parent = model

        # Navigate to parent module
        for part in path_parts[:-1]:
            parent = getattr(parent, part)

        # Replace the final module
        setattr(parent, path_parts[-1], new_module)

    def calculate_compression_stats(self, tier_masks: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Calculate detailed compression statistics"""
        total_neurons = 0
        fp16_neurons = 0
        int4_neurons = 0
        int1_neurons = 0

        layer_stats = {}

        for name, masks in tier_masks.items():
            layer_total = len(masks['fp16'])
            layer_fp16 = masks['fp16'].sum().item()
            layer_int4 = masks['int4'].sum().item()
            layer_int1 = masks['int1'].sum().item()

            total_neurons += layer_total
            fp16_neurons += layer_fp16
            int4_neurons += layer_int4
            int1_neurons += layer_int1

            # Calculate per-layer stats with OneBit overhead
            int1_overhead = 1.0073 if layer_int1 > 0 else 1  # OneBit paper value
            layer_effective_bits = (
                                           layer_fp16 * 16 +
                                           layer_int4 * 4 +
                                           layer_int1 * int1_overhead
                                   ) / layer_total if layer_total > 0 else 16

            layer_compression = 16 / layer_effective_bits if layer_effective_bits > 0 else 1.0

            layer_stats[name] = {
                'total_neurons': layer_total,
                'fp16_neurons': layer_fp16,
                'int4_neurons': layer_int4,
                'int1_neurons': layer_int1,
                'effective_bits': layer_effective_bits,
                'compression_ratio': layer_compression,
                'fp16_percentage': (layer_fp16 / layer_total * 100) if layer_total > 0 else 0,
                'int4_percentage': (layer_int4 / layer_total * 100) if layer_total > 0 else 0,
                'int1_percentage': (layer_int1 / layer_total * 100) if layer_total > 0 else 0,
            }

        # Calculate overall stats with OneBit overhead
        if total_neurons > 0:
            int1_overhead = 1.0073  # OneBit paper value
            effective_bits = (
                                     fp16_neurons * 16 +
                                     int4_neurons * 4 +
                                     int1_neurons * int1_overhead
                             ) / total_neurons
            compression_ratio = 16 / effective_bits
        else:
            effective_bits = 16.0
            compression_ratio = 1.0

        # Calculate memory savings estimate
        original_memory_mb = total_neurons * 16 / 8 / 1024 / 1024  # 16-bit weights in MB
        compressed_memory_mb = (
                                       fp16_neurons * 16 +
                                       int4_neurons * 4 +
                                       int1_neurons * int1_overhead
                               ) / 8 / 1024 / 1024
        memory_savings_mb = original_memory_mb - compressed_memory_mb
        memory_savings_percentage = (memory_savings_mb / original_memory_mb * 100) if original_memory_mb > 0 else 0

        stats = {
            'total_neurons': total_neurons,
            'fp16_neurons': fp16_neurons,
            'int4_neurons': int4_neurons,
            'int1_neurons': int1_neurons,
            'effective_bits': effective_bits,
            'compression_ratio': compression_ratio,
            'fp16_percentage': (fp16_neurons / total_neurons * 100) if total_neurons > 0 else 0,
            'int4_percentage': (int4_neurons / total_neurons * 100) if total_neurons > 0 else 0,
            'int1_percentage': (int1_neurons / total_neurons * 100) if total_neurons > 0 else 0,
            'original_memory_mb': original_memory_mb,
            'compressed_memory_mb': compressed_memory_mb,
            'memory_savings_mb': memory_savings_mb,
            'memory_savings_percentage': memory_savings_percentage,
            'layer_stats': layer_stats,
            'num_converted_layers': len(tier_masks)
        }

        logger.info(f"📊 Compression Statistics:")
        logger.info(f"  • Total neurons: {total_neurons:,}")
        logger.info(f"  • FP16: {fp16_neurons:,} ({stats['fp16_percentage']:.1f}%)")
        logger.info(f"  • 4-bit: {int4_neurons:,} ({stats['int4_percentage']:.1f}%)")
        logger.info(f"  • 1-bit: {int1_neurons:,} ({stats['int1_percentage']:.1f}%)")
        logger.info(f"  • Effective bits: {effective_bits:.2f}")
        logger.info(f"  • Compression: {compression_ratio:.2f}x")
        logger.info(f"  • Memory savings: {memory_savings_mb:.1f} MB ({memory_savings_percentage:.1f}%)")

        return stats
