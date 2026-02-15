#!/usr/bin/env python3
"""Test module for adaptive quantization"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup test logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress debug logs during testing
    format='%(levelname)s - %(name)s - %(message)s'
)

# Test configuration
TEST_CONFIG = {
    'batch_size': 1,
    'num_samples': 8,
    'seq_length': 32,
    'vocab_size': 1000,
    'test_device': 'cpu'  # Use CPU for consistent testing
}

__all__ = ['TEST_CONFIG']
