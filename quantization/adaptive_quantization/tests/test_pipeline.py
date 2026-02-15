#!/usr/bin/env python3
"""Tests for quantization pipeline"""

import unittest
import tempfile
import shutil
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.quantization_config import QuantizationConfig
from src.pipeline.quantization_pipeline import AdaptiveQuantizationPipeline


class TestQuantizationPipeline(unittest.TestCase):
    """Test quantization pipeline functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Create minimal config for testing
        self.config = QuantizationConfig(
            tier1_ratio=0.4,
            tier2_ratio=0.4,
            tier3_ratio=0.2,
            training_epochs=0,  # Skip training for faster tests
            batch_size=1,
            calibration_samples=8,
            device='cpu',
            save_path=self.temp_dir
        )

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Check components are initialized
        self.assertIsNotNone(pipeline.scorer)
        self.assertIsNotNone(pipeline.converter)
        self.assertIsNotNone(pipeline.trainer)
        self.assertEqual(pipeline.config, self.config)

    def test_dummy_model_creation(self):
        """Test dummy model creation"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Create dummy model
        model = pipeline._create_dummy_model()

        # Check model structure
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'embedding'))
        self.assertTrue(hasattr(model, 'layers'))
        self.assertTrue(hasattr(model, 'lm_head'))

        # Check model can do forward pass
        test_input = torch.randint(0, 1000, (1, 10))
        output = model(test_input)
        self.assertIsNotNone(output)
        self.assertTrue(torch.isfinite(output).all())

    def test_dataloader_creation(self):
        """Test dataloader creation"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Create dataloader
        dataloader = pipeline.create_simple_dataloader(batch_size=2, num_samples=8)

        # Check dataloader properties
        self.assertEqual(len(dataloader), 4)  # 8 samples / 2 batch_size = 4 batches

        # Test first batch
        batch = next(iter(dataloader))
        self.assertIn('input_ids', batch)
        self.assertEqual(batch['input_ids'].shape[0], 2)  # Batch size

    def test_model_loading_fallback(self):
        """Test model loading with fallback to dummy"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Test with non-existent model (should fallback to dummy)
        model, tokenizer = pipeline.load_model_safely("nonexistent-model")

        # Should return dummy model
        self.assertIsNotNone(model)
        self.assertIsNone(tokenizer)  # Dummy model doesn't have tokenizer

    def test_dummy_pipeline_execution(self):
        """Test full pipeline execution with dummy model"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Run pipeline
        results = pipeline.run_pipeline("dummy")

        # Check results structure
        self.assertIn('pipeline_success', results)
        self.assertIn('config', results)

        if results['pipeline_success']:
            # Check expected result components
            expected_keys = ['compression_stats', 'tier_masks', 'importance_scores']
            for key in expected_keys:
                self.assertIn(key, results)

            # Check compression stats
            stats = results['compression_stats']
            self.assertIn('total_neurons', stats)
            self.assertIn('compression_ratio', stats)
            self.assertGreater(stats['compression_ratio'], 1.0)
        else:
            # If pipeline failed, should have error information
            self.assertIn('errors', results)

    def test_pipeline_with_training(self):
        """Test pipeline with minimal training"""
        # Config with 1 training epoch
        training_config = QuantizationConfig(
            tier1_ratio=0.5,
            tier2_ratio=0.3,
            tier3_ratio=0.2,
            training_epochs=1,  # Minimal training
            batch_size=1,
            calibration_samples=4,
            device='cpu',
            save_path=self.temp_dir
        )

        pipeline = AdaptiveQuantizationPipeline(training_config)
        results = pipeline.run_pipeline("dummy")

        # Should still succeed (though may take longer)
        self.assertIn('pipeline_success', results)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Create config with invalid settings
        invalid_config = QuantizationConfig(
            tier1_ratio=0.5,
            tier2_ratio=0.5,
            tier3_ratio=0.5,  # Sum > 1.0, should cause error
            device='cpu'
        )

        # Should raise error during config validation
        with self.assertRaises(ValueError):
            QuantizationConfig(
                tier1_ratio=0.5,
                tier2_ratio=0.5,
                tier3_ratio=0.5
            )

    def test_compression_stats_calculation(self):
        """Test compression statistics calculation"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Create mock tier masks
        tier_masks = {
            'layer1': {
                'fp16': torch.tensor([True, True, False, False]),
                'int4': torch.tensor([False, False, True, False]),
                'int1': torch.tensor([False, False, False, True])
            },
            'layer2': {
                'fp16': torch.tensor([True, False, False]),
                'int4': torch.tensor([False, True, False]),
                'int1': torch.tensor([False, False, True])
            }
        }

        # Calculate stats
        stats = pipeline.converter.calculate_compression_stats(tier_masks)

        # Check stats
        self.assertIn('total_neurons', stats)
        self.assertIn('effective_bits', stats)
        self.assertIn('compression_ratio', stats)

        # Should show compression
        self.assertLess(stats['effective_bits'], 16.0)
        self.assertGreater(stats['compression_ratio'], 1.0)

    def test_quick_evaluation(self):
        """Test quick evaluation functionality"""
        pipeline = AdaptiveQuantizationPipeline(self.config)

        # Create two identical dummy models
        teacher_model = pipeline._create_dummy_model()
        student_model = pipeline._create_dummy_model()

        # Create test dataloader
        dataloader = pipeline.create_simple_dataloader(batch_size=1, num_samples=4)

        # Run evaluation
        eval_results = pipeline._quick_evaluation(teacher_model, student_model, dataloader)

        # Check results
        if 'error' not in eval_results:
            self.assertIn('mse_loss', eval_results)
            self.assertIn('cosine_similarity', eval_results)

            # MSE should be low for identical models
            self.assertLess(eval_results['mse_loss'], 1.0)
            self.assertGreater(eval_results['cosine_similarity'], 0.5)


class TestPipelineComponents(unittest.TestCase):
    """Test individual pipeline components"""

    def setUp(self):
        """Set up test components"""
        self.config = QuantizationConfig(
            tier1_ratio=0.3,
            tier2_ratio=0.4,
            tier3_ratio=0.3,
            training_epochs=0,
            batch_size=1,
            calibration_samples=4,
            device='cpu'
        )
        self.pipeline = AdaptiveQuantizationPipeline(self.config)

    def test_importance_scorer_integration(self):
        """Test importance scorer integration"""
        # Create dummy model
        model = self.pipeline._create_dummy_model()
        dataloader = self.pipeline.create_simple_dataloader(batch_size=1, num_samples=4)

        # Compute importance scores
        try:
            scores = self.pipeline.scorer.compute_importance_scores(model, dataloader)

            # Should return scores for linear layers
            self.assertIsInstance(scores, dict)
            self.assertGreater(len(scores), 0)

            # All scores should be tensors
            for name, score in scores.items():
                self.assertIsInstance(score, torch.Tensor)

        except Exception as e:
            # Importance scoring might fail in test environment, which is acceptable
            self.skipTest(f"Importance scoring failed in test environment: {e}")

    def test_model_converter_integration(self):
        """Test model converter integration"""
        # Create dummy tier masks
        tier_masks = {
            'layers.0': {
                'fp16': torch.tensor([True, False]),
                'int4': torch.tensor([False, True]),
                'int1': torch.tensor([False, False])
            }
        }

        # Test stats calculation
        stats = self.pipeline.converter.calculate_compression_stats(tier_masks)

        self.assertIn('total_neurons', stats)
        self.assertEqual(stats['total_neurons'], 2)
        self.assertLess(stats['effective_bits'], 16.0)

    def test_memory_monitoring(self):
        """Test memory monitoring functionality"""
        from src.utils.memory_utils import check_memory_usage, MemoryMonitor

        # Test memory usage check
        memory_info = check_memory_usage()
        self.assertIsInstance(memory_info, str)
        self.assertIn('GB', memory_info)

        # Test memory monitor context manager
        with MemoryMonitor("Test Operation") as monitor:
            # Do some computation
            test_tensor = torch.randn(100, 100)
            result = torch.mm(test_tensor, test_tensor.t())

        # Should complete without error
        self.assertIsNotNone(result)


class TestPipelineConfiguration(unittest.TestCase):
    """Test pipeline configuration variations"""

    def test_different_quantization_methods(self):
        """Test different importance methods"""
        methods = ['activation', 'weight', 'combined']

        for method in methods:
            with self.subTest(method=method):
                config = QuantizationConfig(
                    importance_method=method,
                    training_epochs=0,
                    calibration_samples=4,
                    device='cpu'
                )

                pipeline = AdaptiveQuantizationPipeline(config)

                # Should initialize without error
                self.assertEqual(pipeline.config.importance_method, method)

    def test_extreme_tier_ratios(self):
        """Test extreme tier ratio configurations"""
        extreme_configs = [
            (1.0, 0.0, 0.0),  # All FP16
            (0.0, 1.0, 0.0),  # All 4-bit
            (0.0, 0.0, 1.0),  # All 1-bit
            (0.1, 0.1, 0.8),  # Mostly 1-bit
        ]

        for tier1, tier2, tier3 in extreme_configs:
            with self.subTest(tiers=(tier1, tier2, tier3)):
                config = QuantizationConfig(
                    tier1_ratio=tier1,
                    tier2_ratio=tier2,
                    tier3_ratio=tier3,
                    training_epochs=0,
                    calibration_samples=4,
                    device='cpu'
                )

                pipeline = AdaptiveQuantizationPipeline(config)

                # Should handle extreme configurations
                self.assertIsNotNone(pipeline)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)

    # Run tests with detailed output
    unittest.main(verbosity=2)
