"""
Integration tests for the TensorFlow 2.19.0 fault injection migration

This script validates that the migrated components work correctly together
and maintains compatibility with the original fault injection functionality.
"""

import os
import sys
import tempfile
import tensorflow as tf
import numpy as np
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from injection.injection_types import InjType, is_bit_flip, is_random_value
from injection.injection_args import SimulationParameters, read_injection_config
from injection.injection_utils import (
    binary_to_float32, float32_to_binary, choose_inj_pos
)
from models.resnet_18 import create_resnet18
from building_blocks.residual_block import ResidualBlock
from building_blocks.inject_layers import InjectableConv2D
from tools.utils import load_cifar10_data, setup_tensorflow_config


def test_injection_types():
    """Test injection type system."""
    print("Testing injection types...")
    
    # Test basic type checking
    assert is_bit_flip(InjType.RBFLIP) == True
    assert is_random_value(InjType.RD) == True
    assert is_bit_flip(InjType.RD) == False
    
    print("✓ Injection types working correctly")


def test_binary_conversion():
    """Test binary-float conversion utilities."""
    print("Testing binary conversion...")
    
    test_value = 3.14159
    binary = float32_to_binary(test_value)
    converted_back = binary_to_float32(binary)
    
    assert len(binary) == 32
    assert abs(test_value - converted_back) < 1e-6
    
    print("✓ Binary conversion working correctly")


def test_config_parsing():
    """Test configuration file parsing."""
    print("Testing config parsing...")
    
    # Create a test config file
    test_config = '''
model,resnet18
stage,fwrd_inject
fmodel,RD
target_worker,1
target_layer,basicblock_2_basic_0_conv2
target_epoch,19
target_step,38
inj_pos,107/3/3/85
inj_values,2.86871514043344e+24
learning_rate,0.001
seed,123
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(test_config)
        temp_file = f.name
    
    try:
        params = read_injection_config(temp_file)
        assert params.model == 'resnet18'
        assert params.target_epoch == 19
        assert params.target_step == 38
        assert params.learning_rate == 0.001
        assert len(params.inj_pos) == 1
        assert len(params.inj_values) == 1
        
        print("✓ Config parsing working correctly")
    finally:
        os.unlink(temp_file)


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    # Setup TensorFlow
    setup_tensorflow_config()
    
    # Create model
    model = create_resnet18(seed=123)
    
    # Test forward pass
    sample_input = tf.random.normal((2, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outputs = model(sample_input, training=False)
    
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, NUM_CLASSES)
    
    print("✓ Model creation and forward pass working correctly")


def test_residual_block():
    """Test residual block functionality."""
    print("Testing residual blocks...")
    
    block = ResidualBlock(filters=64, stride=1, seed=123)
    
    # Test forward pass
    sample_input = tf.random.normal((2, 32, 32, 64))
    output, layer_inputs, layer_weights, layer_outputs = block(
        sample_input, training=False
    )
    
    assert output.shape == sample_input.shape
    assert len(layer_inputs) > 0
    assert len(layer_outputs) > 0
    
    print("✓ Residual blocks working correctly")


def test_injectable_layers():
    """Test injectable layer functionality."""
    print("Testing injectable layers...")
    
    layer = InjectableConv2D(
        filters=32, 
        kernel_size=3, 
        padding='same',
        seed=123
    )
    
    # Build layer
    sample_input = tf.random.normal((2, 32, 32, 3))
    output, conv_output = layer(sample_input, training=False)
    
    assert output.shape == (2, 32, 32, 32)
    assert conv_output.shape == (2, 32, 32, 32)
    
    print("✓ Injectable layers working correctly")


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    train_dataset, test_dataset, train_count, test_count = load_cifar10_data(seed=123)
    
    assert train_count == 50000
    assert test_count == 10000
    
    # Test that we can iterate through the dataset
    sample_batch = next(iter(train_dataset.batch(32)))
    images, labels = sample_batch
    
    assert images.shape == (32, 32, 32, 3)
    assert labels.shape == (32,)
    
    print("✓ Data loading working correctly")


def test_injection_position_selection():
    """Test injection position selection."""
    print("Testing injection position selection...")
    
    # Create a test tensor
    test_tensor = np.random.randn(10, 10, 32)
    
    # Create mock injection args
    class MockInjArgs:
        def __init__(self):
            self.inj_pos = []
            self.inj_values = []
    
    inj_args = MockInjArgs()
    
    # Test position selection
    mask, delta = choose_inj_pos(
        test_tensor, 
        InjType.RD, 
        inj_args
    )
    
    assert mask.shape == test_tensor.shape
    assert delta.shape == test_tensor.shape
    assert len(inj_args.inj_pos) > 0
    assert len(inj_args.inj_values) > 0
    
    print("✓ Injection position selection working correctly")


def test_tensorflow_compatibility():
    """Test TensorFlow 2.19.0 compatibility."""
    print("Testing TensorFlow compatibility...")
    
    # Check TensorFlow version
    tf_version = tf.__version__
    print(f"TensorFlow version: {tf_version}")
    
    # Test modern APIs
    tf.keras.utils.set_random_seed(123)
    
    # Test data augmentation layers
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=123),
        tf.keras.layers.RandomCrop(32, 32, seed=123)
    ])
    
    sample_input = tf.random.normal((4, 36, 36, 3))
    augmented = augmentation(sample_input)
    
    assert augmented.shape == (4, 32, 32, 3)
    
    print("✓ TensorFlow 2.19.0 compatibility verified")


def run_all_tests():
    """Run all integration tests."""
    print("Running TensorFlow 2.19.0 Fault Injection Migration Tests\n")
    print("=" * 60)
    
    tests = [
        test_injection_types,
        test_binary_conversion,
        test_config_parsing,
        test_tensorflow_compatibility,
        test_data_loading,
        test_model_creation,
        test_residual_block,
        test_injectable_layers,
        test_injection_position_selection,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Migration is successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the migration.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
