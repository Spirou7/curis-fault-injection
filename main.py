"""
Main training orchestration script for fault injection experiments with TensorFlow 2.19.0

This script provides the main entry point for running fault injection experiments,
including training, fault injection, and result logging.
"""

import os
import argparse
import math
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple

from config import *
from tools.utils import (
    setup_tensorflow_config, load_cifar10_data, prepare_datasets,
    create_learning_rate_schedule, get_optimizer, record
)
from models.resnet_18 import create_resnet18
from injection.injection_args import read_injection_config, log_injection_info
from injection.injection_types import InjType
from injection.injection_utils import choose_inj_pos


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="TensorFlow 2.19.0 Fault Injection Framework"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to injection configuration CSV file"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=OUTPUT_DIR,
        help="Output directory for logs and results"
    )
    parser.add_argument(
        '--mixed_precision', 
        action='store_true',
        help="Enable mixed precision training"
    )
    parser.add_argument(
        '--xla_compile', 
        action='store_true',
        help="Enable XLA compilation"
    )
    parser.add_argument(
        '--use_tpu', 
        action='store_true',
        help="Use TPU for training (if available)"
    )
    
    return parser.parse_args()


def setup_strategy(use_tpu: bool = False) -> tf.distribute.Strategy:
    """
    Setup distributed training strategy.
    
    Args:
        use_tpu: Whether to use TPU
        
    Returns:
        Distribution strategy
    """

    strategy = None

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("TPU is running:", tpu.master())
    except ValueError as e:
            print("TPU is not avaible:", e)
    
    return strategy


def create_model_and_optimizer(strategy: tf.distribute.Strategy, 
                               params: Any) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
    """
    Create model and optimizer within distribution strategy scope.
    
    Args:
        strategy: Distribution strategy
        params: Simulation parameters
        
    Returns:
        Tuple of (model, optimizer)
    """
    with strategy.scope():
        # Create model
        model = create_resnet18(seed=params.seed)
        
        # Create learning rate schedule
        if 'sgd' in params.model.lower():
            lr_schedule = create_learning_rate_schedule(
                initial_lr=params.learning_rate,
                decay_steps=2000,
                end_lr=0.001
            )
            optimizer = get_optimizer('sgd', lr_schedule)
        else:
            lr_schedule = create_learning_rate_schedule(
                initial_lr=params.learning_rate,
                decay_steps=5000,
                end_lr=0.0001
            )
            optimizer = get_optimizer('adam', lr_schedule)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        # Initialize optimizer variables by running a dummy forward pass
        # This ensures all variables are created before TPU compilation
        dummy_input = tf.zeros((1, 32, 32, 3))
        dummy_labels = tf.zeros((1,), dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            dummy_outputs = model(dummy_input, training=True)
            dummy_logits = dummy_outputs['logits']
            dummy_loss = tf.keras.losses.sparse_categorical_crossentropy(dummy_labels, dummy_logits)
        
        # Initialize optimizer variables
        dummy_gradients = tape.gradient(dummy_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(dummy_gradients, model.trainable_variables))
    
    return model, optimizer


@tf.function
def train_step(strategy: tf.distribute.Strategy, 
               model: tf.keras.Model, 
               optimizer: tf.keras.optimizers.Optimizer,
               inputs: tf.Tensor, 
               labels: tf.Tensor) -> tf.Tensor:
    """
    Single training step.
    
    Args:
        strategy: Distribution strategy
        model: Model to train
        optimizer: Optimizer
        inputs: Input batch
        labels: Label batch
        
    Returns:
        Loss value
    """
    def step_fn(inputs, labels):
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            logits = outputs['logits']
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
            loss = tf.nn.compute_average_loss(loss, global_batch_size=BATCH_SIZE)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    return strategy.run(step_fn, args=(inputs, labels))


@tf.function
def test_step(strategy: tf.distribute.Strategy,
              model: tf.keras.Model,
              inputs: tf.Tensor,
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Single test step.
    
    Args:
        strategy: Distribution strategy
        model: Model to evaluate
        inputs: Input batch
        labels: Label batch
        
    Returns:
        Tuple of (loss, accuracy)
    """
    def step_fn(inputs, labels):
        outputs = model(inputs, training=False)
        logits = outputs['logits']
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=VALID_BATCH_SIZE)
        
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, logits)
        accuracy = tf.reduce_mean(accuracy)
        
        return loss, accuracy
    
    return strategy.run(step_fn, args=(inputs, labels))


def load_checkpoint(model: tf.keras.Model, checkpoint_path: str) -> None:
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
    """
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")


def run_training_with_injection(args: argparse.Namespace) -> None:
    """
    Main training loop with fault injection.
    
    Args:
        args: Command line arguments
    """
    # Setup TensorFlow configuration
    setup_tensorflow_config(
        mixed_precision=args.mixed_precision,
        xla_compile=args.xla_compile,
        memory_growth=MEMORY_GROWTH
    )
    
    # Setup distributed strategy
    strategy = setup_strategy(args.use_tpu)
    
    # Load injection configuration
    params = read_injection_config(args.config)
    print(f"Loaded injection config: {params.model}, epoch {params.target_epoch}, step {params.target_step}")
    
    # Load and prepare data
    train_dataset, test_dataset, train_count, test_count = load_cifar10_data(params.seed)
    
    per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync
    train_dataset, test_dataset = prepare_datasets(
        train_dataset, test_dataset, BATCH_SIZE, VALID_BATCH_SIZE, params.seed
    )
    
    # Distribute datasets
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)
    
    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(strategy, params)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    config_name = os.path.basename(args.config).replace('.csv', '')
    log_file = os.path.join(args.output_dir, f"replay_{config_name}.txt")
    
    with open(log_file, 'w') as recorder:
        log_injection_info(recorder, f"Starting injection experiment\n")
        log_injection_info(recorder, f"Target: epoch {params.target_epoch}, step {params.target_step}\n")
        log_injection_info(recorder, f"Model: {params.model}, Layer: {params.target_layer}\n")
        log_injection_info(recorder, f"Fault type: {params.fmodel}\n")
        
        # Load checkpoint if specified
        if hasattr(params, 'checkpoint_path') and params.checkpoint_path:
            load_checkpoint(model, params.checkpoint_path)
        elif os.path.exists(GOLDEN_MODEL_DIR):
            checkpoint_path = os.path.join(
                GOLDEN_MODEL_DIR, params.model, f"epoch_{params.target_epoch - 1}"
            )
            load_checkpoint(model, checkpoint_path)
        
        # Training loop
        steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
        test_steps_per_epoch = math.ceil(test_count / VALID_BATCH_SIZE)
        
        start_epoch = params.target_epoch
        current_epoch = start_epoch
        early_terminate = False
        
        train_iterator = iter(train_dataset)
        
        # Main training loop
        while current_epoch < EPOCHS and not early_terminate:
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            
            # Training steps for this epoch
            for step in range(steps_per_epoch):
                if early_terminate:
                    break
                
                # Get next batch
                try:
                    batch_inputs, batch_labels = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataset)
                    batch_inputs, batch_labels = next(train_iterator)
                
                # Check if this is the injection step
                if current_epoch == params.target_epoch and step == params.target_step:
                    log_injection_info(recorder, f"Performing injection at epoch {current_epoch}, step {step}\n")
                    
                    # TODO: Implement actual fault injection here
                    # This would involve:
                    # 1. Getting layer outputs
                    # 2. Applying injection
                    # 3. Continuing with modified values
                    
                    # For now, just perform normal training step
                    loss = train_step(strategy, model, optimizer, batch_inputs, batch_labels)
                else:
                    # Normal training step
                    loss = train_step(strategy, model, optimizer, batch_inputs, batch_labels)
                
                # Reduce loss across replicas
                step_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                epoch_train_loss += step_loss
                
                # Log step information
                log_injection_info(
                    recorder, 
                    f"Epoch {current_epoch}/{EPOCHS}, Step {step}/{steps_per_epoch}, Loss: {step_loss:.5f}\n"
                )
                
                # Check for NaN
                if not tf.math.is_finite(step_loss):
                    log_injection_info(recorder, "Encountered NaN! Terminating training!\n")
                    early_terminate = True
                    break
            
            if not early_terminate:
                # Validation
                epoch_test_loss = 0.0
                epoch_test_acc = 0.0
                
                test_iterator = iter(test_dataset)
                for _ in range(test_steps_per_epoch):
                    try:
                        test_inputs, test_labels = next(test_iterator)
                        test_loss, test_acc = test_step(strategy, model, test_inputs, test_labels)
                        
                        test_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, test_loss, axis=None)
                        test_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, test_acc, axis=None)
                        
                        epoch_test_loss += test_loss
                        epoch_test_acc += test_acc
                    except StopIteration:
                        break
                
                # Average over steps
                epoch_train_loss /= steps_per_epoch
                epoch_test_loss /= test_steps_per_epoch
                epoch_test_acc /= test_steps_per_epoch
                
                log_injection_info(
                    recorder,
                    f"End of epoch {current_epoch}: "
                    f"train_loss={epoch_train_loss:.5f}, "
                    f"test_loss={epoch_test_loss:.5f}, "
                    f"test_acc={epoch_test_acc:.5f}\n"
                )
                
                # Check for NaN in validation
                if not tf.math.is_finite(epoch_test_loss):
                    log_injection_info(recorder, "Encountered NaN in validation! Terminating training!\n")
                    early_terminate = True
            
            current_epoch += 1
        
        log_injection_info(recorder, "Training completed\n")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    tf.keras.utils.set_random_seed(123)
    
    # Run training with injection
    run_training_with_injection(args)


if __name__ == '__main__':
    main()
