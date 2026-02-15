#!/usr/bin/env python3
"""Training module for adaptive quantization with knowledge distillation"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging
from tqdm import tqdm

from config.quantization_config import QuantizationConfig
from .loss_functions import AdaptiveQuantizationLoss
from ..utils.memory_utils import MemoryMonitor, clear_memory_cache

logger = logging.getLogger(__name__)


class AdaptiveQuantizationTrainer:
    """Trainer for 3-tier adaptive quantization with knowledge distillation"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.loss_fn = AdaptiveQuantizationLoss(
            alpha=config.kd_alpha,
            temperature=config.kd_temperature
        )

    def setup_optimizer_groups(self, student_model: nn.Module) -> List[Dict]:
        """Setup different learning rates for different precision tiers"""
        fp16_params = []
        int4_params = []
        int1_params = []
        other_params = []

        for name, param in student_model.named_parameters():
            if not param.requires_grad:
                continue

            if 'fp16_weight' in name:
                fp16_params.append(param)
            elif any(x in name for x in ['int4_scales', 'int4_zeros']):
                int4_params.append(param)
            elif any(x in name for x in ['int1_input_scale', 'int1_output_scale']):
                int1_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if fp16_params:
            param_groups.append({
                'params': fp16_params,
                'lr': self.config.learning_rate * 0.1,  # Lower LR for stable weights
                'name': 'fp16_params'
            })
            logger.info(f"FP16 parameters: {len(fp16_params)} tensors, lr={self.config.learning_rate * 0.1}")

        if int4_params:
            param_groups.append({
                'params': int4_params,
                'lr': self.config.learning_rate,
                'name': 'int4_params'
            })
            logger.info(f"4-bit parameters: {len(int4_params)} tensors, lr={self.config.learning_rate}")

        if int1_params:
            param_groups.append({
                'params': int1_params,
                'lr': self.config.learning_rate * 1.5,  # Higher LR for OneBit scaling factors
                'name': 'int1_params'
            })
            logger.info(f"1-bit parameters: {len(int1_params)} tensors, lr={self.config.learning_rate * 1.5}")

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate * 0.5,
                'name': 'other_params'
            })
            logger.info(f"Other parameters: {len(other_params)} tensors, lr={self.config.learning_rate * 0.5}")

        if not param_groups:
            raise ValueError("No trainable parameters found!")

        return param_groups

    def train(self, student_model: nn.Module, teacher_model: nn.Module,
              train_dataloader) -> nn.Module:
        """Main training loop with knowledge distillation (OneBit methodology)"""
        logger.info("Starting 3-tier adaptive quantization training...")
        logger.info(f"Training for {self.config.training_epochs} epochs")

        # Setup optimizer
        param_groups = self.setup_optimizer_groups(student_model)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.training_epochs, eta_min=1e-6
        )

        # Training loop
        student_model.train()
        teacher_model.eval()

        training_history = {
            'epoch_losses': [],
            'batch_losses': [],
            'learning_rates': []
        }

        for epoch in range(self.config.training_epochs):
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training_epochs}",
                leave=True
            )

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Process batch input
                    if isinstance(batch, dict):
                        if 'input_ids' in batch:
                            inputs = batch['input_ids'].to(self.config.device)
                        else:
                            inputs = list(batch.values())[0].to(self.config.device)
                    else:
                        inputs = batch.to(self.config.device)

                    # Skip if batch is too small
                    if inputs.size(0) < 1:
                        continue

                    # Forward pass with teacher (no gradients)
                    with torch.no_grad():
                        teacher_outputs = MemoryMonitor(teacher_model, inputs)

                    # Forward pass with student
                    student_outputs = MemoryMonitor(student_model, inputs)

                    # Compute loss
                    loss = self.loss_fn(student_outputs, teacher_outputs)

                    # Check for NaN/Inf
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss detected: {loss}")
                        continue

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        self.config.gradient_clipping
                    )

                    optimizer.step()

                    # Update statistics
                    loss_value = loss.item()
                    epoch_loss += loss_value
                    num_batches += 1

                    training_history['batch_losses'].append(loss_value)

                    # Update progress bar
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'Loss': f'{loss_value:.4f}',
                        'Avg': f'{epoch_loss / num_batches:.4f}',
                        'LR': f'{current_lr:.2e}'
                    })

                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        clear_memory_cache()

                except Exception as e:
                    logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
                    continue

            # End of epoch
            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            training_history['epoch_losses'].append(avg_epoch_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            logger.info(f"Epoch {epoch + 1}/{self.config.training_epochs} completed. "
                        f"Average Loss: {avg_epoch_loss:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping check (simple version)
            if len(training_history['epoch_losses']) > 3:
                recent_losses = training_history['epoch_losses'][-3:]
                if avg_epoch_loss > training_history['epoch_losses'][-2] * 1.1:
                    logger.warning(f"Loss increased significantly. Consider early stopping.")

        logger.info("✅ Training completed successfully!")
        logger.info(f"Final average loss: {training_history['epoch_losses'][-1]:.4f}")

        return student_model
