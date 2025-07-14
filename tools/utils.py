"""
Utility functions for fault injection framework with TensorFlow 2.19.0

This module provides general utility functions including:
- Logging and recording utilities
- Data processing helpers
- File I/O utilities
- Legacy compatibility functions
"""

import struct
import tensorflow as tf
import numpy as np
from typing import Any, Optional, Tuple
from tensorflow.keras.datasets import cifar10


def record(recorder: Any, message: str) -> None:
    """
    Record a message to a file or logger.
    
    Args:
        recorder: File handle or logger object
        message: Message to record
    """
    if recorder:
        recorder.write(message)
        recorder.flush()


def binary_to_float32(binary_string: str) -> float:
    """Convert 32-bit binary string to IEEE 754 float32."""
    if len(binary_string) != 32:
        raise ValueError(f"Binary string must be 32 bits, got {len(binary_string)}")
    return struct.unpack('!f', struct.pack('!I', int(binary_string, 2)))[0]


def float32_to_binary(value: float) -> str:
    """Convert IEEE 754 float32 to 32-bit binary string."""
    return ''.join(format(c, '08b') for c in struct.pack('!f', value))


def normalize_data(train_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize training and test data using training statistics.
    
    Args:
        train_data: Training data array
        test_data: Test data array
        
    Returns:
        Tuple of (normalized_train_data, normalized_test_data)
    """
    mean = np.mean(train_data, axis=(0, 1, 2, 3))
    std = np.std(train_data, axis=(0, 1, 2, 3))
    
    normalized_train = (train_data - mean) / std
    normalized_test = (test_data - mean) / std
    
    return normalized_train, normalized_test


def load_cifar10_data(seed: int = 123) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Load and preprocess CIFAR-10 dataset with TensorFlow 2.19.0 APIs.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset, train_count, test_count)
    """
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize data
    x_train, x_test = normalize_data(x_train.astype(np.float32), x_test.astype(np.float32))
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_dataset, test_dataset, len(x_train), len(x_test)


def prepare_datasets(train_dataset: tf.data.Dataset, 
                    test_dataset: tf.data.Dataset,
                    batch_size: int = 1024,
                    test_batch_size: int = 1000,
                    seed: int = 123) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare datasets for training with modern TensorFlow 2.19.0 APIs.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Training batch size
        test_batch_size: Test batch size
        seed: Random seed
        
    Returns:
        Tuple of (prepared_train_dataset, prepared_test_dataset)
    """
    # Prepare training dataset
    train_dataset = (train_dataset
                    .shuffle(buffer_size=50000, seed=seed)
                    .repeat()
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    
    # Prepare test dataset
    test_dataset = (test_dataset
                   .batch(test_batch_size)
                   .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, test_dataset


def setup_tensorflow_config(mixed_precision: bool = False, 
                           xla_compile: bool = False, 
                           memory_growth: bool = True) -> None:
    """
    Configure TensorFlow 2.19.0 settings for optimal performance.
    
    Args:
        mixed_precision: Enable mixed precision training
        xla_compile: Enable XLA compilation
        memory_growth: Enable GPU memory growth
    """
    # Configure mixed precision
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # Configure XLA
    if xla_compile:
        tf.config.optimizer.set_jit(True)
    
    # Configure GPU memory growth
    if memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs have been initialized
                pass


def create_learning_rate_schedule(initial_lr: float, 
                                 decay_steps: int, 
                                 end_lr: float) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Create a polynomial decay learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        decay_steps: Number of steps to decay over
        end_lr: Final learning rate
        
    Returns:
        Learning rate schedule
    """
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr
    )


def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """
    Get an optimizer by name with TensorFlow 2.19.0 compatibility.
    
    Args:
        name: Optimizer name ('adam', 'sgd', etc.)
        learning_rate: Learning rate or schedule
        
    Returns:
        Optimizer instance
    """
    name = name.lower()
    
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'adamw':
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def multi_tensor_reduce_max(tensor_list: list) -> tf.Tensor:
    """
    Compute the maximum absolute value across multiple tensors.
    
    Args:
        tensor_list: List of tensors
        
    Returns:
        Maximum absolute value across all tensors
    """
    max_values = [tf.reduce_max(tf.abs(tensor)) for tensor in tensor_list]
    return tf.reduce_max(max_values)


def create_checkpoint_manager(model: tf.keras.Model, 
                             optimizer: tf.keras.optimizers.Optimizer,
                             checkpoint_dir: str,
                             max_to_keep: int = 5) -> tf.train.CheckpointManager:
    """
    Create a checkpoint manager for model saving.
    
    Args:
        model: Keras model
        optimizer: Optimizer
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Checkpoint manager
    """
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, 
        checkpoint_dir, 
        max_to_keep=max_to_keep
    )
    return manager


# Legacy compatibility aliases
bin2fp32 = binary_to_float32
fp322bin = float32_to_binary
