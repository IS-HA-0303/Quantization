#!/usr/bin/env python3
"""Data loading and processing utilities"""

import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def create_dummy_dataloader(batch_size: int = 8, num_samples: int = 128,
                            vocab_size: int = 1000, seq_length: int = 512) -> DataLoader:
    """Create a dummy dataloader for testing and calibration"""

    try:
        # Generate random input IDs (token sequences)
        input_ids = torch.randint(1, vocab_size, (num_samples, seq_length))

        # Create attention masks (all ones for simplicity)
        attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

        # Create dataset and dataloader
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"✅ Created dummy dataloader: {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader

    except Exception as e:
        logger.error(f"❌ Failed to create dummy dataloader: {e}")
        # Fallback to minimal dataloader
        return _create_minimal_dataloader(batch_size, num_samples)


def _create_minimal_dataloader(batch_size: int, num_samples: int) -> DataLoader:
    """Create minimal fallback dataloader"""
    input_ids = torch.randint(1, 100, (num_samples, 32))
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"✅ Created minimal fallback dataloader: {len(dataloader)} batches")
    return dataloader


def create_synthetic_language_data(num_samples: int = 1000, seq_length: int = 512,
                                   vocab_size: int = 30000) -> DataLoader:
    """Create synthetic language modeling data"""

    try:
        # Generate more realistic token sequences
        input_ids = []

        for _ in range(num_samples):
            # Start with special tokens
            seq = [1]  # BOS token

            # Generate sequence with some patterns
            for i in range(seq_length - 2):
                # Simple pattern: higher probability for lower token IDs
                if torch.rand(1) < 0.7:
                    token = torch.randint(2, min(1000, vocab_size), (1,)).item()
                else:
                    token = torch.randint(1000, vocab_size, (1,)).item()
                seq.append(token)

            seq.append(2)  # EOS token
            input_ids.append(seq)

        input_ids = torch.tensor(input_ids)

        # Create labels (shifted input for language modeling)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token in loss

        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        logger.info(f"✅ Created synthetic language data: {len(dataset)} samples")
        return dataloader

    except Exception as e:
        logger.error(f"❌ Failed to create synthetic data: {e}")
        return create_dummy_dataloader()


def validate_dataloader(dataloader: DataLoader) -> bool:
    """Validate that dataloader works correctly"""

    try:
        # Try to get first batch
        first_batch = next(iter(dataloader))

        if isinstance(first_batch, (list, tuple)):
            if len(first_batch) == 0:
                return False

            # Check first tensor
            first_tensor = first_batch[0]
            if not isinstance(first_tensor, torch.Tensor):
                return False

            if first_tensor.numel() == 0:
                return False

        elif isinstance(first_batch, dict):
            if len(first_batch) == 0:
                return False

        else:
            return False

        logger.info(f"✅ Dataloader validation passed: {len(dataloader)} batches")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Dataloader validation failed: {e}")
        return False


def prepare_batch(batch, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Prepare batch for model input"""

    try:
        if isinstance(batch, dict):
            # Already in dict format
            prepared = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prepared[key] = value.to(device)
                else:
                    prepared[key] = value
            return prepared

        elif isinstance(batch, (list, tuple)):
            # Convert list/tuple to dict
            prepared = {}
            if len(batch) >= 1:
                prepared['input_ids'] = batch[0].to(device)
            if len(batch) >= 2:
                prepared['attention_mask'] = batch[1].to(device)
            if len(batch) >= 3:
                prepared['labels'] = batch[2].to(device)
            return prepared

        else:
            # Single tensor
            return {'input_ids': batch.to(device)}

    except Exception as e:
        logger.error(f"❌ Failed to prepare batch: {e}")
        return {'input_ids': torch.randint(1, 100, (1, 32)).to(device)}


def calculate_dataset_stats(dataloader: DataLoader) -> Dict[str, Any]:
    """Calculate statistics about the dataset"""

    total_samples = 0
    total_tokens = 0
    max_length = 0
    min_length = float('inf')

    try:
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('inputs', None))
            else:
                input_ids = batch

            if input_ids is not None:
                batch_size, seq_len = input_ids.shape[:2]
                total_samples += batch_size
                total_tokens += batch_size * seq_len
                max_length = max(max_length, seq_len)
                min_length = min(min_length, seq_len)

        stats = {
            'total_samples': total_samples,
            'total_tokens': total_tokens,
            'max_sequence_length': max_length,
            'min_sequence_length': min_length,
            'avg_tokens_per_sample': total_tokens / max(total_samples, 1),
            'num_batches': len(dataloader)
        }

        logger.info(f"📊 Dataset stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"❌ Failed to calculate dataset stats: {e}")
        return {'error': str(e)}
