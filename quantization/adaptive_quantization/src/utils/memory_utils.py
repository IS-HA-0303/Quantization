#!/usr/bin/env python3
"""Memory monitoring utilities"""

import psutil
import torch
import logging
import gc
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations"""

    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_memory = None
        self.peak_memory = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        """Enter the runtime context"""
        gc.collect()  # Clean up before monitoring
        self.start_memory = self._get_memory_usage()
        self.logger.info(f"🔄 Starting {self.operation_name} - Memory: {self.start_memory:.1f}MB")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context"""
        gc.collect()  # Clean up after operation
        end_memory = self._get_memory_usage()
        memory_diff = end_memory - self.start_memory

        if memory_diff > 0:
            self.logger.info(f"✅ {self.operation_name} completed - Memory: {end_memory:.1f}MB (+{memory_diff:.1f}MB)")
        else:
            self.logger.info(f"✅ {self.operation_name} completed - Memory: {end_memory:.1f}MB ({memory_diff:.1f}MB)")

        return False  # Don't suppress exceptions

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0


def clear_memory_cache():
    """Clear PyTorch and Python memory caches"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("🧹 Memory cache cleared")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clear memory cache: {e}")


def check_memory_usage() -> str:
    """Get current system memory usage as string"""
    try:
        memory = psutil.virtual_memory()
        return f"System: {memory.used / 1024 ** 3:.1f}/{memory.total / 1024 ** 3:.1f}GB ({memory.percent:.1f}%)"
    except Exception as e:
        logger.warning(f"⚠️ Failed to check memory usage: {e}")
        return "Memory info unavailable"


def get_gpu_memory_usage() -> Optional[str]:
    """Get GPU memory usage if available"""
    if not torch.cuda.is_available():
        return None

    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)

        return (f"GPU: {gpu_allocated / 1024 ** 3:.1f}GB allocated, "
                f"{gpu_cached / 1024 ** 3:.1f}GB cached, "
                f"{gpu_memory / 1024 ** 3:.1f}GB total")
    except Exception as e:
        logger.warning(f"⚠️ Failed to get GPU memory: {e}")
        return None
