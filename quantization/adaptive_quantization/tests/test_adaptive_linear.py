#!/usr/bin/env python3
"""Tests for adaptive linear layer"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.quantization.adaptive_linear import AdaptivePrecisionLinear


class TestAdaptiveLinear(unittest.TestCase):
    """Test adaptive linear layer functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.in_features = 8
        self.out_features = 4
        self.device = 'cpu'

        # Create tier masks for testing
        self.tier_masks = {
            'fp16': torch.tensor([True, True, False, False]),
            'int4': torch.tensor([False, False, True, False]),
            'int1': torch.tensor([False, False, False, True])
        }

        # Create test input
        self.test_input = torch.randn(2, self.in_features)

    def test_layer_initialization(self):
        """Test layer initialization"""
        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, self.tier_masks, self.device
        )

        # Check parameters exist
        self.assertIsNotNone(layer.fp16_weight)
        self.assertIsNotNone(layer.int4_weight)
        self.assertIsNotNone(layer.int1_sign)
        self.assertIsNotNone(layer.bias)

        # Check shapes
        self.assertEqual(layer.fp16_weight.shape[0], 2)  # 2 FP16 neurons
        self.assertEqual(layer.fp16_weight.shape[1], self.in_features)
        self.assertEqual(layer.bias.shape[0], self.out_features)

    def test_forward_pass(self):
        """Test forward pass"""
        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, self.tier_masks, self.device
        )

        # Forward pass
        output = layer(self.test_input)

        # Check output shape
        self.assertEqual(output.shape, (2, self.out_features))

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_init_from_fp16(self):
        """Test initialization from FP16 weights"""
        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, self.tier_masks, self.device
        )

        # Create original weight
        original_weight = torch.randn(self.out_features, self.in_features)

        # Initialize from FP16
        layer.init_from_fp16(original_weight)

        # Check FP16 weights are set correctly
        fp16_indices = torch.where(self.tier_masks['fp16'])[0]
        expected_fp16_weights = original_weight[fp16_indices]
        self.assertTrue(torch.allclose(layer.fp16_weight, expected_fp16_weights, atol=1e-6))

    def test_compression_stats(self):
        """Test compression statistics calculation"""
        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, self.tier_masks, self.device
        )

        stats = layer.get_compression_stats()

        # Check stats structure
        required_keys = [
            'total_neurons', 'fp16_neurons', 'int4_neurons', 'int1_neurons',
            'effective_bits', 'compression_ratio'
        ]
        for key in required_keys:
            self.assertIn(key, stats)

        # Check values make sense
        self.assertEqual(stats['total_neurons'], self.out_features)
        self.assertEqual(stats['fp16_neurons'], 2)
        self.assertEqual(stats['int4_neurons'], 1)
        self.assertEqual(stats['int1_neurons'], 1)
        self.assertGreater(stats['compression_ratio'], 1.0)
        self.assertLess(stats['effective_bits'], 16.0)

    def test_tier_masks_validation(self):
        """Test tier masks validation"""
        # Test with invalid masks (wrong size)
        invalid_masks = {
            'fp16': torch.tensor([True, False]),  # Wrong size
            'int4': torch.tensor([False, True]),
            'int1': torch.tensor([False, False])
        }

        with self.assertRaises((RuntimeError, ValueError)):
            layer = AdaptivePrecisionLinear(
                self.in_features, self.out_features, invalid_masks, self.device
            )

    def test_gradients_flow(self):
        """Test that gradients flow correctly"""
        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, self.tier_masks, self.device
        )

        # Enable gradients
        self.test_input.requires_grad = True

        # Forward pass
        output = layer(self.test_input)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(self.test_input.grad)
        if layer.fp16_weight is not None:
            self.assertIsNotNone(layer.fp16_weight.grad)

    def test_different_tier_configurations(self):
        """Test different tier configurations"""
        configs = [
            # All FP16
            {
                'fp16': torch.tensor([True, True, True, True]),
                'int4': torch.tensor([False, False, False, False]),
                'int1': torch.tensor([False, False, False, False])
            },
            # All 1-bit
            {
                'fp16': torch.tensor([False, False, False, False]),
                'int4': torch.tensor([False, False, False, False]),
                'int1': torch.tensor([True, True, True, True])
            },
            # Mixed configuration
            {
                'fp16': torch.tensor([True, False, True, False]),
                'int4': torch.tensor([False, True, False, False]),
                'int1': torch.tensor([False, False, False, True])
            }
        ]

        for i, config in enumerate(configs):
            with self.subTest(f"Config {i}"):
                layer = AdaptivePrecisionLinear(
                    self.in_features, self.out_features, config, self.device
                )

                output = layer(self.test_input)
                self.assertEqual(output.shape, (2, self.out_features))
                self.assertTrue(torch.isfinite(output).all())

    def test_layer_with_zero_tiers(self):
        """Test behavior when some tiers have zero neurons"""
        zero_tier_masks = {
            'fp16': torch.tensor([True, True, True, True]),
            'int4': torch.tensor([False, False, False, False]),  # No 4-bit neurons
            'int1': torch.tensor([False, False, False, False])  # No 1-bit neurons
        }

        layer = AdaptivePrecisionLinear(
            self.in_features, self.out_features, zero_tier_masks, self.device
        )

        # Should still work with only FP16
        output = layer(self.test_input)
        self.assertEqual(output.shape, (2, self.out_features))

        # Check that unused tier parameters are None
        self.assertIsNone(layer.int4_weight)
        self.assertIsNone(layer.int1_sign)


class TestAdaptiveLinearEdgeCases(unittest.TestCase):
    """Test edge cases for adaptive linear layer"""

    def test_single_neuron_layers(self):
        """Test with single neuron per tier"""
        tier_masks = {
            'fp16': torch.tensor([True]),
            'int4': torch.tensor([False]),
            'int1': torch.tensor([False])
        }

        layer = AdaptivePrecisionLinear(4, 1, tier_masks, 'cpu')
        test_input = torch.randn(1, 4)

        output = layer(test_input)
        self.assertEqual(output.shape, (1, 1))

    def test_large_layer_simulation(self):
        """Test with larger layer dimensions"""
        in_features, out_features = 64, 32

        # Create random tier assignment
        total_neurons = out_features
        fp16_count = total_neurons // 3
        int4_count = total_neurons // 3
        int1_count = total_neurons - fp16_count - int4_count

        tier_masks = {
            'fp16': torch.cat([
                torch.ones(fp16_count, dtype=torch.bool),
                torch.zeros(total_neurons - fp16_count, dtype=torch.bool)
            ]),
            'int4': torch.cat([
                torch.zeros(fp16_count, dtype=torch.bool),
                torch.ones(int4_count, dtype=torch.bool),
                torch.zeros(int1_count, dtype=torch.bool)
            ]),
            'int1': torch.cat([
                torch.zeros(fp16_count + int4_count, dtype=torch.bool),
                torch.ones(int1_count, dtype=torch.bool)
            ])
        }

        layer = AdaptivePrecisionLinear(in_features, out_features, tier_masks, 'cpu')
        test_input = torch.randn(4, in_features)

        output = layer(test_input)
        self.assertEqual(output.shape, (4, out_features))

        # Check compression is achieved
        stats = layer.get_compression_stats()
        self.assertLess(stats['effective_bits'], 16.0)
        self.assertGreater(stats['compression_ratio'], 1.0)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)  # For reproducible tests

    # Run tests
    unittest.main(verbosity=2)
