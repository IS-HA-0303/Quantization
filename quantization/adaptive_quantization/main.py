#!/usr/bin/env python3
"""
3-Tier Adaptive Quantization System
Based on OneBit paper methodology with importance-based neuron ranking
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.quantization_config import QuantizationConfig
from src.pipeline.quantization_pipeline import AdaptiveQuantizationPipeline


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('quantization.log')
        ]
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3-Tier Adaptive Quantization for LLMs')

    # Model configuration
    parser.add_argument('--model-name', type=str, default='dummy',
                        help='Model name or path (default: dummy for testing)')

    # Quantization tiers
    parser.add_argument('--tier1-ratio', type=float, default=0.2,
                        help='Ratio of neurons for Tier 1 (FP16) (default: 0.2)')
    parser.add_argument('--tier2-ratio', type=float, default=0.5,
                        help='Ratio of neurons for Tier 2 (4-bit) (default: 0.5)')
    parser.add_argument('--tier3-ratio', type=float, default=0.3,
                        help='Ratio of neurons for Tier 3 (1-bit) (default: 0.3)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    # Data parameters
    parser.add_argument('--calibration-samples', type=int, default=64,
                        help='Number of calibration samples (default: 64)')

    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto) (default: auto)')
    parser.add_argument('--save-path', type=str, default='./models/quantized',
                        help='Path to save quantized model (default: ./models/quantized)')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    # Check tier ratios sum to 1.0
    total_ratio = args.tier1_ratio + args.tier2_ratio + args.tier3_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Tier ratios must sum to 1.0, got {total_ratio}")

    # Check positive values
    if args.tier1_ratio <= 0 or args.tier2_ratio <= 0 or args.tier3_ratio <= 0:
        raise ValueError("All tier ratios must be positive")

    if args.epochs <= 0:
        raise ValueError("Epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if args.calibration_samples <= 0:
        raise ValueError("Calibration samples must be positive")


def main():
    """Main entry point"""
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting 3-Tier Adaptive Quantization System")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Validate arguments
        validate_args(args)

        # Create configuration
        config = QuantizationConfig(
            tier1_ratio=args.tier1_ratio,
            tier2_ratio=args.tier2_ratio,
            tier3_ratio=args.tier3_ratio,
            training_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            calibration_samples=args.calibration_samples,
            device=args.device,
            save_path=args.save_path
        )

        logger.info("✅ Configuration created successfully")
        logger.info(f"📊 Config: {config}")

        # Initialize and run pipeline
        pipeline = AdaptiveQuantizationPipeline(config)
        logger.info("✅ Pipeline initialized")

        # Run quantization
        results = pipeline.run_pipeline(args.model_name)

        # Display results
        if results and results.get('pipeline_success', False):
            logger.info("🎉 Pipeline completed successfully!")

            if 'compression_stats' in results:
                stats = results['compression_stats']
                logger.info("📊 Compression Results:")
                logger.info(f"  • Total neurons: {stats.get('total_neurons', 0):,}")
                logger.info(f"  • Effective bits: {stats.get('effective_bits', 16):.2f}")
                logger.info(f"  • Compression ratio: {stats.get('compression_ratio', 1):.2f}x")
                logger.info(f"  • Memory savings: {stats.get('memory_savings_mb', 0):.1f} MB")

            if 'evaluation' in results:
                eval_results = results['evaluation']
                if 'mse_loss' in eval_results:
                    logger.info("🎯 Quality Metrics:")
                    logger.info(f"  • MSE Loss: {eval_results['mse_loss']:.6f}")
                    logger.info(f"  • Cosine Similarity: {eval_results['cosine_similarity']:.4f}")

            logger.info(f"💾 Model saved to: {config.save_path}")
            return 0
        else:
            logger.error("❌ Pipeline failed!")
            if 'errors' in results:
                for error in results['errors']:
                    logger.error(f"  • {error}")
            return 1

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
