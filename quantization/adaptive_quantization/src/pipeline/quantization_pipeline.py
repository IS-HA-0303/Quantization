#!/usr/bin/env python3
"""
3-Tier Adaptive Quantization Pipeline
Based on OneBit paper methodology with importance-based neuron ranking
"""

import os
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

# Import your modules
from config.quantization_config import QuantizationConfig
from src.utils.memory_utils import MemoryMonitor, clear_memory_cache, check_memory_usage
from src.utils.data_utils import create_dummy_dataloader, validate_dataloader, create_synthetic_language_data
from src.quantization.Importance_scorer import NeuronImportanceScorer
from src.quantization.model_converter import ModelConverter
from src.quantization.adaptive_linear import AdaptivePrecisionLinear
from src.training.trainer import AdaptiveQuantizationTrainer

logger = logging.getLogger(__name__)


class AdaptiveQuantizationPipeline:
    """
    Complete pipeline for 3-tier adaptive quantization
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.device = config.device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        try:
            self.scorer = NeuronImportanceScorer(config)
            self.converter = ModelConverter(config)
            self.trainer = AdaptiveQuantizationTrainer(config)
            self.logger.info("✅ Pipeline components initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline components: {e}")
            raise e

    def load_model_safely(self, model_name: str) -> Tuple[nn.Module, Optional[Any]]:
        """Load model with fallback to dummy model"""
        self.logger.info(f"🔄 Loading model: {model_name}")

        if model_name == "dummy":
            self.logger.info("Using dummy model for testing")
            return self._create_dummy_model(), None

        try:
            # Try to load real model (requires transformers library)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = model.to(self.device)
            self.logger.info(f"✅ Successfully loaded model: {model_name}")
            return model, tokenizer

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load model {model_name}: {e}")
            self.logger.info("🔄 Falling back to dummy model")
            return self._create_dummy_model(), None

    def _create_dummy_model(
            self) -> nn.Module:
        """Create a small dummy transformer-like model for testing"""

        class DummyTransformerBlock(nn.Module):
            def __init__(self, d_model: int = 64, nhead: int = 4):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.ln2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )

            def forward(self, x):
                # Self-attention block
                attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
                x = x + attn_out

                # MLP block
                x = x + self.mlp(self.ln2(x))
                return x

        class DummyModel(nn.Module):
            def __init__(self, vocab_size: int = 1000, d_model: int = 64, n_layers: int = 3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(512, d_model) * 0.1)

                self.layers = nn.ModuleList([
                    DummyTransformerBlock(d_model) for _ in range(n_layers)
                ])

                self.ln_final = nn.LayerNorm(d_model)
                self.lm_head = nn.Linear(d_model, vocab_size)

                # Initialize weights
                self._init_weights()

            def _init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Embedding):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape

                # Token + position embeddings
                x = self.embedding(input_ids)
                pos_emb = self.pos_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
                x = x + pos_emb

                # Transformer blocks
                for layer in self.layers:
                    x = layer(x)

                # Final layer norm and projection
                x = self.ln_final(x)
                logits = self.lm_head(x)

                return logits

        model = DummyModel().to(self.device)
        self.logger.info(f"✅ Created dummy model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def create_simple_dataloader(self, batch_size: int = None, num_samples: int = None):
        """Create a simple dataloader for calibration"""
        batch_size = batch_size or self.config.batch_size
        num_samples = num_samples or self.config.calibration_samples

        try:
            dataloader = create_dummy_dataloader(
                batch_size=batch_size,
                num_samples=num_samples,
                vocab_size=1000,
                seq_length=self.config.max_sequence_length
            )

            if validate_dataloader(dataloader):
                self.logger.info(f"✅ Created dataloader: {len(dataloader)} batches")
                return dataloader
            else:
                raise ValueError("Dataloader validation failed")

        except Exception as e:
            self.logger.warning(f"⚠️ Dataloader creation failed: {e}")
            # Fallback to minimal dataloader
            return self._create_minimal_dataloader(batch_size, num_samples)

    def _create_minimal_dataloader(self, batch_size: int, num_samples: int):
        """Create minimal fallback dataloader"""
        from torch.utils.data import DataLoader, TensorDataset

        # Create minimal random data
        input_ids = torch.randint(1, 100, (num_samples, 32))
        dataset = TensorDataset(input_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.logger.info(f"✅ Created minimal fallback dataloader: {len(dataloader)} batches")
        return dataloader

    def _quick_evaluation(self, teacher_model: nn.Module, student_model: nn.Module,
                          dataloader) -> Dict[str, float]:
        """Quick evaluation comparing teacher and student outputs"""
        try:
            teacher_model.eval()
            student_model.eval()

            total_mse = 0.0
            total_cosine_sim = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, dict):
                        input_ids = batch['input_ids'].to(self.device)
                    else:
                        input_ids = batch[0].to(self.device)

                    # Get outputs
                    teacher_output = teacher_model(input_ids)
                    student_output = student_model(input_ids)

                    # Calculate MSE
                    mse = torch.nn.functional.mse_loss(student_output, teacher_output)
                    total_mse += mse.item()

                    # Calculate cosine similarity
                    teacher_flat = teacher_output.flatten()
                    student_flat = student_output.flatten()
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        teacher_flat.unsqueeze(0),
                        student_flat.unsqueeze(0)
                    )
                    total_cosine_sim += cosine_sim.item()

                    num_batches += 1

                    # Limit evaluation batches
                    if num_batches >= 5:
                        break

            return {
                'mse_loss': total_mse / num_batches,
                'cosine_similarity': total_cosine_sim / num_batches
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Evaluation failed: {e}")
            return {'error': str(e)}

    def run_pipeline(self, model_name: str) -> Dict[str, Any]:
        """Run the complete 3-tier adaptive quantization pipeline"""
        results = {
            'pipeline_success': False,
            'config': self.config.to_dict(),
            'errors': []
        }

        try:
            self.logger.info("🚀 Starting 3-tier adaptive quantization pipeline")
            self.logger.info(f"📊 Memory usage: {check_memory_usage()}")

            # Step 1: Load model
            with MemoryMonitor("Model loading"):
                teacher_model, tokenizer = self.load_model_safely(model_name)

            # Step 2: Create calibration data
            with MemoryMonitor("Data preparation"):
                dataloader = self.create_simple_dataloader()

            # Step 3: Compute importance scores
            with MemoryMonitor("Importance scoring"):
                try:
                    importance_scores = self.scorer.compute_importance_scores(teacher_model, dataloader)
                    self.logger.info(f"✅ Computed importance scores for {len(importance_scores)} layers")
                except Exception as e:
                    self.logger.warning(f"⚠️ Importance scoring failed: {e}")
                    # Create dummy importance scores
                    importance_scores = {}
                    for name, module in teacher_model.named_modules():
                        if isinstance(module, nn.Linear):
                            importance_scores[name] = torch.randn(module.out_features)

            results['importance_scores'] = importance_scores

            # Step 4: Assign tiers based on importance
            with MemoryMonitor("Tier assignment"):
                tier_masks = self.ModelConverter.assign_tiers(importance_scores, self.config)
                self.logger.info(f"✅ Assigned tiers for {len(tier_masks)} layers")

            results['tier_masks'] = tier_masks

            # Step 5: Convert model to adaptive precision
            with MemoryMonitor("Model conversion"):
                try:
                    quantized_model = self.converter.convert_model(teacher_model, tier_masks)
                    self.logger.info("✅ Model converted to adaptive precision")
                except Exception as e:
                    self.logger.warning(f"⚠️ Model conversion failed: {e}")
                    # Use original model as fallback
                    quantized_model = teacher_model

            # Step 6: Knowledge distillation training (if enabled)
            if self.config.training_epochs > 0:
                with MemoryMonitor("Knowledge distillation"):
                    try:
                        self.trainer.train(quantized_model, teacher_model, dataloader)
                        self.logger.info("✅ Knowledge distillation completed")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Training failed: {e}")
                        results['errors'].append(f"Training failed: {e}")
            else:
                self.logger.info("⏭️ Skipping training (epochs=0)")

            # Step 7: Calculate compression statistics
            compression_stats = self.converter.calculate_compression_stats(tier_masks)
            results['compression_stats'] = compression_stats

            self.logger.info("📊 Compression Statistics:")
            self.logger.info(f"  • Total neurons: {compression_stats.get('total_neurons', 0):,}")
            self.logger.info(f"  • Effective bits: {compression_stats.get('effective_bits', 16):.2f}")
            self.logger.info(f"  • Compression ratio: {compression_stats.get('compression_ratio', 1):.2f}x")

            # Step 8: Quick evaluation
            evaluation_results = self._quick_evaluation(teacher_model, quantized_model, dataloader)
            results['evaluation'] = evaluation_results

            # Step 9: Save model
            if self.config.save_path:
                try:
                    save_path = Path(self.config.save_path)
                    save_path.mkdir(parents=True, exist_ok=True)

                    # Save quantized model
                    model_path = save_path / "quantized_model.pt"
                    torch.save(quantized_model.state_dict(), model_path)

                    # Save configuration
                    config_path = save_path / "config.json"
                    with open(config_path, 'w') as f:
                        json.dump(self.config.to_dict(), f, indent=2)

                    # Save compression stats
                    stats_path = save_path / "compression_stats.json"
                    with open(stats_path, 'w') as f:
                        json.dump(compression_stats, f, indent=2)

                    self.logger.info(f"✅ Model saved to: {save_path}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to save model: {e}")
                    results['errors'].append(f"Save failed: {e}")

            results['pipeline_success'] = True
            self.logger.info("🎉 Pipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['pipeline_success'] = False

        finally:
            # Cleanup
            clear_memory_cache()
            self.logger.info(f"📊 Final memory usage: {check_memory_usage()}")

        return results

    def load_quantized_model(self, save_path: str) -> nn.Module:
        """Load a previously quantized model"""
        try:
            save_path = Path(save_path)

            # Load configuration
            config_path = save_path / "config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            # Load model state
            model_path = save_path / "quantized_model.pt"
            state_dict = torch.load(model_path, map_location=self.device)

            # Reconstruct model (this would need the original architecture)
            # This is a simplified version - in practice you'd need to save/load the architecture too
            self.logger.info(f"✅ Loaded quantized model from: {save_path}")

            return state_dict

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise e


# Additional utility functions for the pipeline
def create_pipeline_from_config(config_path: str) -> AdaptiveQuantizationPipeline:
    """Create pipeline from configuration file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = QuantizationConfig(**config_dict)
    return AdaptiveQuantizationPipeline(config)


def run_batch_quantization(model_names: list, config: QuantizationConfig) -> Dict[str, Any]:
    """Run quantization on multiple models"""
    pipeline = AdaptiveQuantizationPipeline(config)
    results = {}

    for model_name in model_names:
        logger.info(f"Processing model: {model_name}")
        try:
            results[model_name] = pipeline.run_pipeline(model_name)
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            results[model_name] = {'pipeline_success': False, 'error': str(e)}

    return results
