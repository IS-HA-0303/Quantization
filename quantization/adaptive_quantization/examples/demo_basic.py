#!/usr/bin/env python3
"""
Basic demo of 3-tier adaptive quantization
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.quantization_config import QuantizationConfig
from src.pipeline.quantization_pipeline import AdaptiveQuantizationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


def main():
    logger = logging.getLogger(__name__)
    logger.info("🔧 Basic 3-Tier Quantization Demo")

    try:
        # Create configuration
        config = QuantizationConfig(
            tier1_ratio=0.3,  # 30% FP16 (high precision)
            tier2_ratio=0.4,  # 40% 4-bit (medium precision)
            tier3_ratio=0.3,  # 30% 1-bit (low precision)
            training_epochs=0,  # Skip training for quick demo
            calibration_samples=16,  # Small sample for demo
            batch_size=1,
            device='cpu',  # Use CPU for compatibility
            save_path="./models/ss"
        )

        logger.info(f"Using device: {config.device}")

        # Initialize pipeline
        pipeline = AdaptiveQuantizationPipeline(config)

        # Run quantization on dummy model
        results = pipeline.run_pipeline("dummy")

        # ✅ SAFE ACCESS TO RESULTS - Check if keys exist before accessing
        if results.get('pipeline_success', False):
            logger.info("🎉 Pipeline completed successfully!")

            # Safely access compression stats
            if 'compression_stats' in results:
                stats = results['compression_stats']
                logger.info("📊 Compression Results:")
                logger.info(f"  • Total neurons: {stats.get('total_neurons', 'N/A'):,}")
                logger.info(f"  • Effective bits: {stats.get('effective_bits', 'N/A'):.2f}")
                logger.info(f"  • Compression ratio: {stats.get('compression_ratio', 'N/A'):.2f}x")
            else:
                logger.info("📊 Compression stats not available")

            # Safely access evaluation results
            if 'evaluation' in results:
                eval_results = results['evaluation']
                if 'error' not in eval_results:
                    logger.info("🎯 Quality Metrics:")
                    logger.info(f"  • MSE Loss: {eval_results.get('mse_loss', 'N/A'):.6f}")
                    logger.info(f"  • Cosine Similarity: {eval_results.get('cosine_similarity', 'N/A'):.4f}")

            # Show save location
            if config.save_path:
                logger.info(f"✅ Model saved to: {config.save_path}")
        else:
            logger.error("❌ Pipeline failed!")
            if 'errors' in results:
                for error in results['errors']:
                    logger.error(f"  • {error}")

            # Still try to show any available stats
            if 'compression_stats' in results:
                logger.info("📊 Partial results available:")
                stats = results['compression_stats']
                for key, value in stats.items():
                    logger.info(f"  • {key}: {value}")

        logger.info("✅ Demo completed successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {'pipeline_success': False, 'error': str(e)}


if __name__ == "__main__":
    results = main()

    # Exit with appropriate code
    if results.get('pipeline_success', False):
        sys.exit(0)
    else:
        sys.exit(1)
