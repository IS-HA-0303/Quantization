#!/usr/bin/env python3
"""Safe tensor operations with error handling"""

import torch
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def safe_tensor_operation(tensor: torch.Tensor, operation: Callable, *args, **kwargs) -> torch.Tensor:
    """
    Safely perform a tensor operation with error handling

    Args:
        tensor: Input tensor
        operation: Operation function to apply
        *args: Additional arguments for the operation
        **kwargs: Additional keyword arguments for the operation

    Returns:
        torch.Tensor: Result of operation or original tensor if operation fails
    """
    try:
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"Input is not a tensor, got {type(tensor)}")
            return tensor

        result = operation(tensor, *args, **kwargs)

        # Validate result
        if not isinstance(result, torch.Tensor):
            logger.warning(f"Operation returned non-tensor: {type(result)}")
            return tensor

        # Check for NaN or Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning("Operation produced NaN or Inf values")
            return tensor

        return result

    except Exception as e:
        logger.error(f"Error during tensor operation {operation.__name__}: {e}")
        return tensor


def safe_tensor_clone(tensor: torch.Tensor) -> torch.Tensor:
    """Safely clone a tensor"""
    return safe_tensor_operation(tensor, torch.clone)


def safe_tensor_detach(tensor: torch.Tensor) -> torch.Tensor:
    """Safely detach a tensor"""
    return safe_tensor_operation(tensor, lambda x: x.detach())


def safe_tensor_to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Safely move tensor to device"""
    return safe_tensor_operation(tensor, lambda x: x.to(device))


def safe_tensor_squeeze(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Safely squeeze tensor"""
    if dim is not None:
        return safe_tensor_operation(tensor, torch.squeeze, dim=dim)
    else:
        return safe_tensor_operation(tensor, torch.squeeze)


def safe_tensor_reshape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Safely reshape tensor"""
    return safe_tensor_operation(tensor, lambda x: x.reshape(shape))


def validate_tensor(tensor: Any) -> bool:
    """Validate if input is a proper tensor"""
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.numel() == 0:
        return False
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    return True


def tensor_info(tensor: torch.Tensor) -> dict:
    """Get tensor information for debugging"""
    if not isinstance(tensor, torch.Tensor):
        return {'error': 'Not a tensor'}

    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'numel': tensor.numel(),
        'memory_mb': tensor.element_size() * tensor.numel() / 1024 / 1024,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min_val': tensor.min().item() if tensor.numel() > 0 else None,
        'max_val': tensor.max().item() if tensor.numel() > 0 else None,
    }
