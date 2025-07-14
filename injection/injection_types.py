"""
Fault injection types and classification utilities for TensorFlow 2.19.0

This module defines the various types of faults that can be injected into neural network
training and provides utility functions for classifying and handling these fault types.
"""

from enum import Enum
from typing import Tuple
import tensorflow as tf


class InjType(Enum):
    """Enumeration of fault injection types."""
    
    # Single-value injections
    INPUT = 1           # Single input activation fault
    INPUT_16 = 2        # 16-bit input activation fault
    WT = 3              # Single weight fault
    WT_16 = 4           # 16-bit weight fault
    RBFLIP = 5          # Random bit flip
    RD = 6              # Random value injection
    RD_CORRECT = 7      # Random correct value injection
    ZERO = 8            # Zero injection
    
    # Multi-value injections
    N16_RD = 9          # 16-position random injection
    N16_RD_CORRECT = 10 # 16-position random correct injection
    RD_GLB = 11         # Global random injection (16 positions, 1000 repetitions)
    RD_CORRECT_GLB = 12 # Global random correct injection
    
    # 64-position injections
    N64_INPUT = 13      # 64-position input injection
    N64_WT = 14         # 64-position weight injection
    N64_INPUT_16 = 15   # 64-position 16-bit input injection
    N64_WT_16 = 16      # 64-position 16-bit weight injection
    N64_INPUT_GLB = 17  # 64-position global input injection
    N64_WT_GLB = 18     # 64-position global weight injection

# Fault classification functions
def is_bit_flip(inj_type: InjType) -> bool:
    """Check if injection type involves bit flipping."""
    return inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16, InjType.RBFLIP]


def is_random_value(inj_type: InjType) -> bool:
    """Check if injection type uses random values."""
    return inj_type in [InjType.RD, InjType.N16_RD, InjType.RD_GLB]


def is_zero_injection(inj_type: InjType) -> bool:
    """Check if injection type sets values to zero."""
    return inj_type in [InjType.ZERO]


def is_correct_value(inj_type: InjType) -> bool:
    """Check if injection type uses correct values from other positions."""
    return inj_type in [
        InjType.RD_CORRECT, InjType.N16_RD_CORRECT, InjType.RD_CORRECT_GLB,
        InjType.N64_INPUT, InjType.N64_WT, InjType.N64_INPUT_16, InjType.N64_WT_16,
        InjType.N64_INPUT_GLB, InjType.N64_WT_GLB
    ]


# Target classification functions
def is_input_target(inj_type: InjType) -> bool:
    """Check if injection targets input activations."""
    return inj_type in [
        InjType.INPUT, InjType.INPUT_16, InjType.N64_INPUT, 
        InjType.N64_INPUT_16, InjType.N64_INPUT_GLB
    ]


def is_weight_target(inj_type: InjType) -> bool:
    """Check if injection targets weights."""
    return inj_type in [
        InjType.WT, InjType.WT_16, InjType.N64_WT, 
        InjType.N64_WT_16, InjType.N64_WT_GLB
    ]


def is_output_target(inj_type: InjType) -> bool:
    """Check if injection targets output activations."""
    return inj_type in [
        InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO,
        InjType.N16_RD, InjType.N16_RD_CORRECT, InjType.RD_GLB, InjType.RD_CORRECT_GLB
    ]


def get_injection_count(inj_type: InjType) -> Tuple[int, int]:
    """
    Get the number of injections and repetitions for a given injection type.
    
    Args:
        inj_type: The injection type
        
    Returns:
        Tuple of (num_injections, num_repetitions)
    """
    if inj_type in [
        InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16,
        InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO
    ]:
        return 1, 1
    elif inj_type in [InjType.N16_RD, InjType.N16_RD_CORRECT]:
        return 16, 1
    elif inj_type in [InjType.RD_GLB, InjType.RD_CORRECT_GLB]:
        return 16, 1000
    elif inj_type in [InjType.N64_INPUT, InjType.N64_WT, InjType.N64_INPUT_16, InjType.N64_WT_16]:
        return 64, 1
    elif inj_type in [InjType.N64_INPUT_GLB, InjType.N64_WT_GLB]:
        return 64, 1000
    else:
        raise ValueError(f"Unknown injection type: {inj_type}")


class InjectionState(Enum):
    """States for injection execution."""
    IDLE = 1
    GOLDEN = 2
    INJECT = 3


# Backward compatibility aliases
is_bflip = is_bit_flip
is_random = is_random_value
is_zero = is_zero_injection
is_correct = is_correct_value
num_inj = get_injection_count
InjState = InjectionState