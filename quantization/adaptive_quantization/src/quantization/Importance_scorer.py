#!/usr/bin/env python3
"""
Importance Scorer for 3-Tier Adaptive Quantization
Computes neuron importance scores based on activations, weights, or combined methods
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Computes importance scores for neurons in linear layers
    Supports multiple methods: activation-based, weight-based, combined
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.method = getattr(config, 'importance_method', 'combined')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Supported methods
        self.supported_methods = ['activation', 'weight', 'combined', 'gradient']

        if self.method not in self.supported_methods:
            self.logger.warning(f"Unknown method '{self.method}', defaulting to 'combined'")
            self.method = 'combined'

        self.logger.info(f"ImportanceScorer initialized with method: {self.method}")

    def compute_importance_scores(self, model: nn.Module, dataloader) -> Dict[str, torch.Tensor]:
        """
        Main method to compute importance scores for all linear layers

        Args:
            model: PyTorch model to analyze
            dataloader: DataLoader with calibration data

        Returns:
            Dictionary mapping layer names to importance score tensors
        """
        self.logger.info(f"Computing importance scores using method: {self.method}")

        try:
            # Set model to evaluation mode
            model.eval()

            # Find all linear layers
            linear_layers = self._find_linear_layers(model)
            self.logger.info(f"Found {len(linear_layers)} linear layers")

            # Initialize importance scores dictionary
            importance_scores = {}

            if self.method == 'activation':
                importance_scores = self._compute_activation_based_scores(model, linear_layers, dataloader)
            elif self.method == 'weight':
                importance_scores = self._compute_weight_based_scores(model, linear_layers)
            elif self.method == 'gradient':
                importance_scores = self._compute_gradient_based_scores(model, linear_layers, dataloader)
            else:  # combined
                importance_scores = self._compute_combined_scores(model, linear_layers, dataloader)

            # Validate results
            self._validate_scores(importance_scores)

            self.logger.info(f"Successfully computed importance scores for {len(importance_scores)} layers")
            return importance_scores

        except Exception as e:
            self.logger.error(f"Error computing importance scores: {e}")
            # Return dummy scores as fallback
            return self._create_dummy_scores(model)

    def _find_linear_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all linear layers in the model"""
        linear_layers = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module

        return linear_layers

    def _compute_weight_based_scores(self, model: nn.Module, linear_layers: Dict[str, nn.Module]) -> Dict[
        str, torch.Tensor]:
        """Compute importance scores based on weight magnitudes"""
        self.logger.info("Computing weight-based importance scores")

        scores = {}

        for name, layer in linear_layers.items():
            try:
                # Get weight matrix
                weight = layer.weight.data  # Shape: (out_features, in_features)

                # Compute L2 norm of each output neuron (row)
                neuron_scores = torch.norm(weight, p=2, dim=1)  # Shape: (out_features,)

                # Normalize scores
                if neuron_scores.sum() > 0:
                    neuron_scores = neuron_scores / neuron_scores.sum()

                scores[name] = neuron_scores.cpu()

            except Exception as e:
                self.logger.warning(f"Error computing weight scores for {name}: {e}")
                scores[name] = torch.ones(layer.out_features) / layer.out_features

        return scores

    def _compute_activation_based_scores(self, model: nn.Module, linear_layers: Dict[str, nn.Module], dataloader) -> \
    Dict[str, torch.Tensor]:
        """Compute importance scores based on activation statistics"""
        self.logger.info("Computing activation-based importance scores")

        # Dictionary to store activation statistics
        activation_stats = {name: [] for name in linear_layers.keys()}

        # Register hooks to capture activations
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                # Store activation statistics
                if isinstance(output, torch.Tensor):
                    # Compute mean absolute activation per neuron
                    activation = output.detach()
                    mean_activation = activation.abs().mean(dim=0)  # Average over batch dimension
                    if len(mean_activation.shape) > 1:
                        mean_activation = mean_activation.flatten()
                    activation_stats[name].append(mean_activation.cpu())

            return hook_fn

        # Register hooks
        for name, layer in linear_layers.items():
            hook = layer.register_forward_hook(make_hook(name))
            hooks.append(hook)

        try:
            # Run forward passes to collect activations
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing activations")):
                    if batch_idx >= self.config.calibration_samples // self.config.batch_size:
                        break

                    # Handle different batch formats
                    if isinstance(batch, dict):
                        inputs = batch['input_ids'].to(self.device)
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)

                    # Forward pass
                    _ = model(inputs)

            # Compute final scores
            scores = {}
            for name in linear_layers.keys():
                if activation_stats[name]:
                    # Average across all batches
                    avg_activations = torch.stack(activation_stats[name]).mean(dim=0)

                    # Normalize
                    if avg_activations.sum() > 0:
                        avg_activations = avg_activations / avg_activations.sum()

                    scores[name] = avg_activations
                else:
                    # Fallback if no activations captured
                    layer = linear_layers[name]
                    scores[name] = torch.ones(layer.out_features) / layer.out_features

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return scores

    def _compute_gradient_based_scores(self, model: nn.Module, linear_layers: Dict[str, nn.Module], dataloader) -> Dict[
        str, torch.Tensor]:
        """Compute importance scores based on gradient magnitudes"""
        self.logger.info("Computing gradient-based importance scores")

        # Enable gradients
        model.train()

        # Dictionary to store gradient statistics
        gradient_stats = {name: [] for name in linear_layers.keys()}

        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradients")):
                if batch_idx >= min(10, len(dataloader)):  # Limit batches for gradient computation
                    break

                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['input_ids'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Forward pass
                model.zero_grad()
                outputs = model(inputs)

                # Create dummy loss (sum of all outputs)
                if isinstance(outputs, torch.Tensor):
                    loss = outputs.sum()
                else:
                    loss = sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))

                # Backward pass
                loss.backward()

                # Collect gradients
                for name, layer in linear_layers.items():
                    if layer.weight.grad is not None:
                        grad_norm = torch.norm(layer.weight.grad, p=2, dim=1)  # Per output neuron
                        gradient_stats[name].append(grad_norm.detach().cpu())

            # Compute final scores
            scores = {}
            for name, layer in linear_layers.items():
                if gradient_stats[name]:
                    # Average gradient norms across batches
                    avg_gradients = torch.stack(gradient_stats[name]).mean(dim=0)

                    # Normalize
                    if avg_gradients.sum() > 0:
                        avg_gradients = avg_gradients / avg_gradients.sum()

                    scores[name] = avg_gradients
                else:
                    # Fallback
                    scores[name] = torch.ones(layer.out_features) / layer.out_features

        except Exception as e:
            self.logger.error(f"Error in gradient computation: {e}")
            # Fallback to weight-based scores
            return self._compute_weight_based_scores(model, linear_layers)

        finally:
            model.eval()  # Reset to eval mode

        return scores

    def _compute_combined_scores(self, model: nn.Module, linear_layers: Dict[str, nn.Module], dataloader) -> Dict[
        str, torch.Tensor]:
        """Compute importance scores using combined weight and activation methods"""
        self.logger.info("Computing combined importance scores")

        # Get weight-based scores
        weight_scores = self._compute_weight_based_scores(model, linear_layers)

        # Get activation-based scores
        try:
            activation_scores = self._compute_activation_based_scores(model, linear_layers, dataloader)
        except Exception as e:
            self.logger.warning(f"Failed to compute activation scores: {e}, using weight scores only")
            return weight_scores

        # Combine scores (weighted average)
        combined_scores = {}
        weight_factor = 0.3
        activation_factor = 0.7

        for name in linear_layers.keys():
            w_score = weight_scores.get(name, torch.zeros(linear_layers[name].out_features))
            a_score = activation_scores.get(name, torch.zeros(linear_layers[name].out_features))

            # Ensure same length
            min_len = min(len(w_score), len(a_score))
            w_score = w_score[:min_len]
            a_score = a_score[:min_len]

            # Combine with weighted average
            combined = weight_factor * w_score + activation_factor * a_score

            # Normalize
            if combined.sum() > 0:
                combined = combined / combined.sum()

            combined_scores[name] = combined

        return combined_scores

    def _validate_scores(self, scores: Dict[str, torch.Tensor]):
        """Validate computed importance scores"""
        for name, score in scores.items():
            if not isinstance(score, torch.Tensor):
                raise ValueError(f"Score for {name} is not a tensor")

            if torch.isnan(score).any():
                self.logger.warning(f"NaN values in scores for {name}, replacing with uniform")
                scores[name] = torch.ones_like(score) / len(score)

            if torch.isinf(score).any():
                self.logger.warning(f"Inf values in scores for {name}, replacing with uniform")
                scores[name] = torch.ones_like(score) / len(score)

    def _create_dummy_scores(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Create dummy importance scores as fallback"""
        self.logger.warning("Creating dummy importance scores")

        dummy_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create uniform importance scores
                uniform_scores = torch.ones(module.out_features) / module.out_features
                dummy_scores[name] = uniform_scores

        return dummy_scores


# Utility functions
def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize importance scores to sum to 1"""
    if scores.sum() > 0:
        return scores / scores.sum()
    else:
        return torch.ones_like(scores) / len(scores)


def compute_percentile_thresholds(scores: torch.Tensor, tier_ratios: tuple) -> List[float]:
    """Compute percentile thresholds for tier assignment"""
    tier1_ratio, tier2_ratio, tier3_ratio = tier_ratios

    # Compute cumulative ratios
    tier1_threshold = 1.0 - tier1_ratio
    tier2_threshold = tier1_threshold - tier2_ratio

    # Get percentile values
    thresholds = [
        torch.quantile(scores, tier1_threshold).item(),
        torch.quantile(scores, tier2_threshold).item()
    ]

    return thresholds


# Export the main class
__all__ = ['ImportanceScorer', 'normalize_scores', 'compute_percentile_thresholds']
