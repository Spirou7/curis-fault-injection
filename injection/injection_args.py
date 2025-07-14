"""
Injection arguments and parameter handling for TensorFlow 2.19.0

This module provides classes and utilities for managing injection experiment parameters,
including injection arguments, simulation parameters, and experiment configuration.
"""

import csv
import dataclasses
from typing import List, Optional, Any, Union
import tensorflow as tf
import numpy as np

from .injection_types import InjType, InjectionState


@dataclasses.dataclass
class InjectionArgs:
    """Arguments for fault injection experiments."""
    
    inj_replica: int                    # Target replica for injection
    inj_layer: str                      # Target layer name
    inj_type: InjType                   # Type of injection
    golden_weights: Union[np.ndarray, List[np.ndarray]]  # Original weights
    golden_output: tf.Tensor            # Original output
    inj_mask: Optional[np.ndarray] = None     # Injection mask
    inj_delta: Optional[np.ndarray] = None    # Injection delta values
    
    def __post_init__(self):
        """Validate injection arguments after initialization."""
        if not isinstance(self.inj_type, InjType):
            raise ValueError(f"Invalid injection type: {self.inj_type}")
        
        if self.inj_replica < 0:
            raise ValueError(f"Invalid replica ID: {self.inj_replica}")


@dataclasses.dataclass
class SimulationParameters:
    """Parameters for fault injection simulation."""
    
    model: str = ''                     # Model name (e.g., 'resnet18')
    stage: str = ''                     # Injection stage ('fwrd_inject' or 'bkwd_inject')
    fmodel: str = ''                    # Fault model (e.g., 'RD', 'ZERO')
    target_worker: int = -1             # Target worker/replica
    target_layer: str = ''              # Target layer name
    target_epoch: int = -1              # Target epoch for injection
    target_step: int = -1               # Target step for injection
    inj_pos: List[List[int]] = dataclasses.field(default_factory=list)    # Injection positions
    inj_values: List[float] = dataclasses.field(default_factory=list)     # Injection values
    learning_rate: float = -1.0         # Learning rate
    seed: int = 123                     # Random seed
    
    def validate(self) -> None:
        """Validate simulation parameters."""
        if not self.model:
            raise ValueError("Model name is required")
        if not self.stage:
            raise ValueError("Stage is required")
        if not self.fmodel:
            raise ValueError("Fault model is required")
        if self.target_epoch < 0:
            raise ValueError("Target epoch must be non-negative")
        if self.target_step < 0:
            raise ValueError("Target step must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


def parse_position_string(pos_string: str) -> List[int]:
    """Parse position string like '107/3/3/85' into list of integers."""
    return [int(x) for x in pos_string.split('/')]


def read_injection_config(file_path: str) -> SimulationParameters:
    """
    Read injection configuration from CSV file.
    
    Args:
        file_path: Path to CSV configuration file
        
    Returns:
        SimulationParameters object with loaded configuration
    """
    params = SimulationParameters()
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
                
            param_name = row[0].strip()
            param_value = row[1].strip()
            
            if param_name == 'model':
                params.model = param_value
            elif param_name == 'stage':
                params.stage = param_value
            elif param_name == 'fmodel':
                params.fmodel = param_value
            elif param_name == 'target_worker':
                params.target_worker = int(param_value)
            elif param_name == 'target_layer':
                params.target_layer = param_value
            elif param_name == 'target_epoch':
                params.target_epoch = int(param_value)
            elif param_name == 'target_step':
                params.target_step = int(param_value)
            elif param_name == 'learning_rate':
                params.learning_rate = float(param_value)
            elif param_name == 'seed':
                params.seed = int(param_value)
            elif param_name == 'inj_pos':
                # Parse multiple positions from remaining columns
                for i in range(1, len(row)):
                    if row[i].strip():
                        params.inj_pos.append(parse_position_string(row[i].strip()))
            elif param_name == 'inj_values':
                # Parse multiple values from remaining columns
                for i in range(1, len(row)):
                    if row[i].strip():
                        params.inj_values.append(float(row[i].strip()))
    
    params.validate()
    return params


def log_injection_info(recorder: Any, message: str) -> None:
    """
    Log injection information to recorder.
    
    Args:
        recorder: File handle or logger object
        message: Message to log
    """
    if recorder:
        recorder.write(message)
        recorder.flush()


# Legacy compatibility class
class InjArgs:
    """Legacy compatibility class for InjArgs."""
    
    def __init__(
        self,
        inj_replica: int,
        inj_layer: str,
        inj_type: InjType,
        golden_weights: Union[np.ndarray, List[np.ndarray]] = None,
        golden_output: tf.Tensor = None,
        mask: Optional[np.ndarray] = None,
        delta: Optional[np.ndarray] = None,
    ):
        """Initialize legacy injection arguments."""
        if not isinstance(inj_type, InjType):
            raise ValueError(f"Invalid injection type: {inj_type}")
        
        self.inj_replica = inj_replica
        self.inj_layer = inj_layer
        self.inj_type = inj_type
        self.golden_weights = golden_weights
        self.golden_output = golden_output
        self.inj_mask = mask
        self.inj_delta = delta