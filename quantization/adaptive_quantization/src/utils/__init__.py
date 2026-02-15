#!/usr/bin/env python3
"""Utils package initialization"""

# Memory utilities
from .memory_utils import (
    MemoryMonitor,
    clear_memory_cache,
    check_memory_usage,
    get_gpu_memory_usage
)

# Data utilities
from .data_utils import (
    create_dummy_dataloader,
    create_synthetic_language_data,
    validate_dataloader
)

# Tensor utilities
from .tensor_utils import (
    safe_tensor_operation,
    safe_tensor_clone,
    safe_tensor_detach,
    safe_tensor_to_device,
    safe_tensor_squeeze,
    safe_tensor_reshape,
    validate_tensor,
    tensor_info
)

__all__ = [
    # Memory utilities
    'MemoryMonitor',
    'clear_memory_cache',
    'check_memory_usage',
    'get_gpu_memory_usage',

    # Data utilities
    'create_dummy_dataloader',
    'create_synthetic_language_data',
    'validate_dataloader',

    # Tensor utilities
    'safe_tensor_operation',
    'safe_tensor_clone',
    'safe_tensor_detach',
    'safe_tensor_to_device',
    'safe_tensor_squeeze',
    'safe_tensor_reshape',
    'validate_tensor',
    'tensor_info'
]
