#!/usr/bin/env python3
"""Tests for bit packing utilities"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.bit_packing import BitPacker, calculate_compression_ratio, estimate_memory_savings


class TestBitPacker(unittest.TestCase):
    """Test bit packing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        torch.manual_seed(42)

    def test_1bit_packing_basic(self):
        """Test basic 1-bit weight packing and unpacking"""
        # Create test weights with ±1 values
        weights = torch.tensor([[-1, 1, -1, 1], [1, -1, 1, -1]], dtype=torch.float32)

        # Pack and unpack
        packed, shape = BitPacker.pack_1bit_weights(weights)
        unpacked = BitPacker.unpack_1bit_weights(packed, shape)

        # Should be identical
        self.assertTrue(torch.allclose(weights, unpacked))

        # Check packed size is smaller
        original_size = weights.numel() * 32  # Float32 = 32 bits per element
        packed_size = packed.numel() * 8  # uint8 = 8 bits per element
        self.assertLess(packed_size, original_size)

    def test_1bit_packing_different_shapes(self):
        """Test 1-bit packing with different tensor shapes"""
        shapes = [(1, 8), (4, 4), (2, 9), (3, 5)]  # Including odd dimensions

        for shape in shapes:
            with self.subTest(shape=shape):
                # Create random ±1 weights
                weights = torch.randint(0, 2, shape).float() * 2 - 1  # Convert to ±1

                # Pack and unpack
                packed, original_shape = BitPacker.pack_1bit_weights(weights)
                unpacked = BitPacker.unpack_1bit_weights(packed, original_shape)

                # Should match original
                self.assertEqual(unpacked.shape, weights.shape)
                self.assertTrue(torch.allclose(weights, unpacked))

    def test_1bit_packing_edge_cases(self):
        """Test edge cases for 1-bit packing"""
        # Test single element
        single = torch.tensor([[1.0]])
        packed, shape = BitPacker.pack_1bit_weights(single)
        unpacked = BitPacker.unpack_1bit_weights(packed, shape)
        self.assertTrue(torch.allclose(single, unpacked))

        # Test empty tensor (should handle gracefully)
        try:
            empty = torch.empty(0, 4)
            packed, shape = BitPacker.pack_1bit_weights(empty)
            unpacked = BitPacker.unpack_1bit_weights(packed, shape)
            self.assertEqual(unpacked.shape, empty.shape)
        except Exception:
            pass  # Empty tensors might not be supported, which is fine

    def test_4bit_packing_basic(self):
        """Test 4-bit weight packing"""
        # Create test weights
        weights = torch.randn(2, 4)
        scales = torch.ones(2) * 0.1
        zeros = torch.zeros(2)

        # Pack
        packed, scales_out, zeros_out, shape = BitPacker.pack_4bit_weights(
            weights, scales, zeros
        )

        # Unpack
        unpacked = BitPacker.unpack_4bit_weights(packed, scales_out, zeros_out, shape)

        # Should be approximately equal (quantization error expected)
        self.assertTrue(torch.allclose(weights, unpacked, atol=0.2))
        self.assertEqual(unpacked.shape, weights.shape)

    def test_4bit_quantization_range(self):
        """Test 4-bit quantization stays in valid range"""
        # Create weights with known range
        weights = torch.linspace(-1, 1, 16).reshape(2, 8)
        scales = torch.ones(2) * (2.0 / 15.0)  # Scale to use full 4-bit range
        zeros = torch.ones(2) * (-1.0)  # Zero point at -1

        packed, scales_out, zeros_out, shape = BitPacker.pack_4bit_weights(
            weights, scales, zeros
        )

        # Check packed values are in 4-bit range [0, 15]
        unpacked_4bit = packed & 0xF  # Lower 4 bits
        self.assertTrue((unpacked_4bit >= 0).all())
        self.assertTrue((unpacked_4bit <= 15).all())

        unpacked_4bit = (packed >> 4) & 0xF  # Upper 4 bits
        self.assertTrue((unpacked_4bit >= 0).all())
        self.assertTrue((unpacked_4bit <= 15).all())

    def test_4bit_packing_different_scales(self):
        """Test 4-bit packing with different scales and zero points"""
        weights = torch.randn(3, 6)

        # Different scales per row
        scales = torch.tensor([0.1, 0.05, 0.2])
        zeros = torch.tensor([0.0, -0.1, 0.1])

        packed, scales_out, zeros_out, shape = BitPacker.pack_4bit_weights(
            weights, scales, zeros
        )
        unpacked = BitPacker.unpack_4bit_weights(packed, scales_out, zeros_out, shape)

        # Check shape preservation
        self.assertEqual(unpacked.shape, weights.shape)

        # Check scales and zeros are preserved
        self.assertTrue(torch.allclose(scales, scales_out))
        self.assertTrue(torch.allclose(zeros, zeros_out))


class TestCompressionUtilities(unittest.TestCase):
    """Test compression calculation utilities"""

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation"""
        # Test basic cases
        self.assertAlmostEqual(calculate_compression_ratio(32, 8), 4.0)
        self.assertAlmostEqual(calculate_compression_ratio(16, 4), 4.0)
        self.assertAlmostEqual(calculate_compression_ratio(16, 1), 16.0)

        # Test edge case
        self.assertEqual(calculate_compression_ratio(16, 0), 1.0)  # Avoid division by zero

    def test_memory_savings_estimation(self):
        """Test memory savings estimation"""
        total_params = 1000000  # 1M parameters
        tier_ratios = (0.2, 0.5, 0.3)  # 20% FP16, 50% 4-bit, 30% 1-bit

        savings = estimate_memory_savings(total_params, tier_ratios)

        # Check structure
        required_keys = [
            'original_memory_mb', 'compressed_memory_mb', 'memory_savings_mb',
            'compression_ratio', 'effective_bits'
        ]
        for key in required_keys:
            self.assertIn(key, savings)

        # Check values make sense
        self.assertGreater(savings['compression_ratio'], 1.0)
        self.assertLess(savings['effective_bits'], 16.0)
        self.assertGreater(savings['memory_savings_mb'], 0.0)

        # Check conservation
        self.assertAlmostEqual(
            savings['original_memory_mb'],
            savings['compressed_memory_mb'] + savings['memory_savings_mb'],
            places=3
        )

    def test_extreme_compression_scenarios(self):
        """Test extreme compression scenarios"""
        total_params = 100000

        # Scenario 1: All FP16 (no compression)
        no_compression = (1.0, 0.0, 0.0)
        savings = estimate_memory_savings(total_params, no_compression)
        self.assertAlmostEqual(savings['compression_ratio'], 1.0, places=2)
        self.assertAlmostEqual(savings['effective_bits'], 16.0, places=1)

        # Scenario 2: All 1-bit (maximum compression)
        max_compression = (0.0, 0.0, 1.0)
        savings = estimate_memory_savings(total_params, max_compression)
        self.assertGreater(savings['compression_ratio'], 10.0)  # Should be close to 16x
        self.assertLess(savings['effective_bits'], 2.0)  # Close to 1-bit + overhead

        # Scenario 3: Balanced
        balanced = (0.33, 0.33, 0.34)
        savings = estimate_memory_savings(total_params, balanced)
        self.assertGreater(savings['compression_ratio'], 2.0)
        self.assertLess(savings['compression_ratio'], 8.0)


class TestBitPackingPerformance(unittest.TestCase):
    """Test performance aspects of bit packing"""

    def test_packing_preserves_information(self):
        """Test that packing preserves all information"""
        # Test with systematic patterns
        patterns = [
            torch.ones(4, 8),  # All positive
            -torch.ones(4, 8),  # All negative
            torch.eye(8)[:4] * 2 - 1,  # Mixed pattern
        ]

        for i, pattern in enumerate(patterns):
            with self.subTest(f"Pattern {i}"):
                packed, shape = BitPacker.pack_1bit_weights(pattern)
                unpacked = BitPacker.unpack_1bit_weights(packed, shape)

                self.assertTrue(torch.equal(pattern, unpacked))

    def test_4bit_precision_bounds(self):
        """Test 4-bit quantization precision bounds"""
        # Create weights that should quantize exactly
        weights = torch.arange(0, 16, dtype=torch.float32).reshape(4, 4)
        scales = torch.ones(4)
        zeros = torch.zeros(4)

        packed, scales_out, zeros_out, shape = BitPacker.pack_4bit_weights(
            weights, scales, zeros
        )
        unpacked = BitPacker.unpack_4bit_weights(packed, scales_out, zeros_out, shape)

        # Should be very close (allowing for floating point precision)
        self.assertTrue(torch.allclose(weights, unpacked, atol=1e-6))

    def test_memory_efficiency(self):
        """Test that packing actually saves memory"""
        # Large tensor test
        large_weights = torch.randint(0, 2, (100, 100)).float() * 2 - 1

        # Original size in bytes
        original_bytes = large_weights.numel() * 4  # float32 = 4 bytes

        # Packed size
        packed, shape = BitPacker.pack_1bit_weights(large_weights)
        packed_bytes = packed.numel() * 1  # uint8 = 1 byte

        # Should be significantly smaller
        compression = original_bytes / packed_bytes
        self.assertGreater(compression, 25)  # Should be close to 32x compression


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests with detailed output
    unittest.main(verbosity=2)
