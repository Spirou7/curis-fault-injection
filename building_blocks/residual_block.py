"""
Residual block implementation for TensorFlow 2.19.0 with fault injection support

This module provides residual block implementations that support:
- TensorFlow 2.19.0 APIs
- Fault injection capabilities
- Modern layer composition
- Proper gradient flow
"""

import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .inject_layers import InjectableConv2D, InjectableBatchNormalization, InjectableReLU
from ..injection.injection_args import InjArgs
from ..injection.injection_types import InjType


class ResidualBlock(tf.keras.layers.Layer):
    """
    ResNet residual block with fault injection support for TensorFlow 2.19.0.
    
    This implementation includes:
    - Skip connections with proper dimension matching
    - Fault injection capabilities
    - Modern TensorFlow 2.19.0 APIs
    - Layer information tracking for injection targeting
    """
    
    def __init__(self, 
                 filters: int,
                 stride: int = 1,
                 seed: Optional[int] = None,
                 dropout_rate: float = 0.0,
                 name: str = 'residual_block',
                 **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        
        self.filters = filters
        self.stride = stride
        self.seed = seed
        self.dropout_rate = dropout_rate
        
        # First convolution block
        self.conv1 = InjectableConv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            use_bias=False,
            seed=seed,
            layer_name=f'{name}_conv1',
            name=f'{name}_conv1'
        )
        
        self.bn1 = InjectableBatchNormalization(
            layer_name=f'{name}_bn1',
            momentum=0.99,  # Updated for TF 2.19.0
            name=f'{name}_bn1'
        )
        
        self.relu1 = InjectableReLU(
            layer_name=f'{name}_relu1',
            name=f'{name}_relu1'
        )
        
        # Dropout layer
        if dropout_rate > 0:
            self.dropout1 = tf.keras.layers.Dropout(
                rate=dropout_rate,
                seed=seed,
                name=f'{name}_dropout1'
            )
        else:
            self.dropout1 = None
        
        # Second convolution block
        self.conv2 = InjectableConv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            seed=seed,
            layer_name=f'{name}_conv2',
            name=f'{name}_conv2'
        )
        
        self.bn2 = InjectableBatchNormalization(
            layer_name=f'{name}_bn2',
            momentum=0.99,  # Updated for TF 2.19.0
            name=f'{name}_bn2'
        )
        
        # Skip connection adjustment if needed
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                InjectableConv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=stride,
                    padding='same',
                    use_bias=False,
                    seed=seed,
                    layer_name=f'{name}_downsample_conv',
                    name=f'{name}_downsample_conv'
                ),
                InjectableBatchNormalization(
                    layer_name=f'{name}_downsample_bn',
                    momentum=0.99,
                    name=f'{name}_downsample_bn'
                )
            ], name=f'{name}_downsample')
        else:
            self.downsample = None
        
        # Final ReLU
        self.relu2 = InjectableReLU(
            layer_name=f'{name}_relu2',
            name=f'{name}_relu2'
        )
        
        # Track layer information for injection
        self._layer_outputs = {}
        self._layer_inputs = {}
        self._layer_weights = {}
    
    def call(self, 
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Forward pass through the residual block.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            inject: Whether to apply fault injection
            inj_args: Injection arguments
            
        Returns:
            Tuple of (output, layer_inputs, layer_weights, layer_outputs)
        """
        # Reset layer tracking
        self._layer_outputs.clear()
        self._layer_inputs.clear()
        self._layer_weights.clear()
        
        # Store initial input
        self._layer_inputs[f'{self.name}_input'] = inputs
        
        # Skip connection
        if self.downsample is not None:
            self._layer_inputs[f'{self.name}_downsample'] = inputs
            residual = self.downsample(inputs, training=training)
            self._layer_outputs[f'{self.name}_downsample'] = residual
        else:
            residual = inputs
        
        # Main path - First convolution block
        self._layer_inputs[f'{self.name}_conv1'] = inputs
        self._layer_weights[f'{self.name}_conv1'] = self.conv1.weights
        x, conv1_out = self.conv1(inputs, training=training, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_conv1'] = conv1_out
        
        # First batch normalization
        self._layer_inputs[f'{self.name}_bn1'] = x
        self._layer_weights[f'{self.name}_bn1'] = self.bn1.weights
        x = self.bn1(x, training=training, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_bn1'] = x
        
        # First ReLU
        self._layer_inputs[f'{self.name}_relu1'] = x
        x = self.relu1(x, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_relu1'] = x
        
        # Dropout if specified
        if self.dropout1 is not None:
            self._layer_inputs[f'{self.name}_dropout1'] = x
            x = self.dropout1(x, training=training)
            self._layer_outputs[f'{self.name}_dropout1'] = x
        
        # Second convolution block
        self._layer_inputs[f'{self.name}_conv2'] = x
        self._layer_weights[f'{self.name}_conv2'] = self.conv2.weights
        x, conv2_out = self.conv2(x, training=training, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_conv2'] = conv2_out
        
        # Second batch normalization
        self._layer_inputs[f'{self.name}_bn2'] = x
        self._layer_weights[f'{self.name}_bn2'] = self.bn2.weights
        x = self.bn2(x, training=training, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_bn2'] = x
        
        # Add skip connection
        self._layer_inputs[f'{self.name}_add'] = (x, residual)
        x = tf.add(x, residual)
        self._layer_outputs[f'{self.name}_add'] = x
        
        # Final ReLU
        self._layer_inputs[f'{self.name}_relu2'] = x
        output = self.relu2(x, inject=inject, inj_args=inj_args)
        self._layer_outputs[f'{self.name}_relu2'] = output
        
        return output, self._layer_inputs, self._layer_weights, self._layer_outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride,
            'seed': self.seed,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ResidualBlock':
        """Create layer from configuration."""
        return cls(**config)


class BasicBlocks(tf.keras.layers.Layer):
    """
    Sequential container for multiple residual blocks.
    
    This is a compatibility layer for the original codebase that
    expects a BasicBlocks layer containing multiple BasicBlock layers.
    """
    
    def __init__(self, 
                 filter_num: int,
                 blocks: int,
                 stride: int = 1,
                 seed: Optional[int] = None,
                 drop_out_rate: float = 0.0,
                 l_name: str = 'basic_blocks',
                 **kwargs):
        super(BasicBlocks, self).__init__(name=l_name, **kwargs)
        
        self.filter_num = filter_num
        self.blocks = blocks
        self.stride = stride
        self.seed = seed
        self.drop_out_rate = drop_out_rate
        self.l_name = l_name
        
        # Create residual blocks
        self.block_layers = []
        for i in range(blocks):
            block_stride = stride if i == 0 else 1  # Only first block has stride > 1
            block = ResidualBlock(
                filters=filter_num,
                stride=block_stride,
                seed=seed,
                dropout_rate=drop_out_rate,
                name=f'{l_name}_basic_{i}'
            )
            self.block_layers.append(block)
    
    def call(self, 
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Forward pass through all residual blocks.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            inject: Whether to apply fault injection
            inj_args: Injection arguments
            
        Returns:
            Tuple of (output, combined_layer_inputs, combined_layer_weights, combined_layer_outputs)
        """
        x = inputs
        
        combined_inputs = {}
        combined_weights = {}
        combined_outputs = {}
        
        for block in self.block_layers:
            x, block_inputs, block_weights, block_outputs = block(
                x, training=training, inject=inject, inj_args=inj_args
            )
            
            # Combine dictionaries
            combined_inputs.update(block_inputs)
            combined_weights.update(block_weights)
            combined_outputs.update(block_outputs)
        
        return x, combined_inputs, combined_weights, combined_outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'filter_num': self.filter_num,
            'blocks': self.blocks,
            'stride': self.stride,
            'seed': self.seed,
            'drop_out_rate': self.drop_out_rate,
            'l_name': self.l_name
        })
        return config


# Legacy compatibility alias
BasicBlock = ResidualBlock