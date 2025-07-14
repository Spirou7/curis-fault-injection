"""
Injection-aware layer implementations for TensorFlow 2.19.0

This module provides custom layer implementations that support fault injection:
- Convolution layers with injection capabilities
- Batch normalization with injection support
- Activation functions with injection hooks
- Dense layers with fault injection
- Utility functions for tensor manipulation
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Any, Union, List, Tuple

from ..config import PER_REPLICA_BATCH_SIZE
from ..injection.injection_types import is_input_target, is_weight_target, is_output_target
from ..injection.injection_args import InjArgs


def apply_injection_to_tensor(tensor: tf.Tensor, inj_args: InjArgs) -> tf.Tensor:
    """
    Apply fault injection to a tensor.
    
    Args:
        tensor: Input tensor
        inj_args: Injection arguments
        
    Returns:
        Tensor with injection applied
    """
    if not is_weight_target(inj_args.inj_type):
        # For activations, apply mask and delta as tensors
        mask_tensor = tf.convert_to_tensor(inj_args.inj_mask, dtype=tf.float32)
        delta_tensor = tf.convert_to_tensor(inj_args.inj_delta, dtype=tf.float32)
        return tf.add(tf.multiply(tensor, mask_tensor), delta_tensor)
    else:
        # For weights, return numpy array with injection applied
        return np.add(
            np.multiply(inj_args.golden_weights[0], inj_args.inj_mask), 
            inj_args.inj_delta
        )


def apply_weight_injection(tensor: tf.Tensor, inj_args: InjArgs) -> tf.Tensor:
    """
    Apply fault injection specifically to weight tensors.
    
    Args:
        tensor: Weight tensor
        inj_args: Injection arguments
        
    Returns:
        Tensor with weight injection applied
    """
    mask_tensor = tf.convert_to_tensor(inj_args.inj_mask, dtype=tf.float32)
    delta_tensor = tf.convert_to_tensor(inj_args.inj_delta, dtype=tf.float32)
    return tf.add(tf.multiply(tensor, mask_tensor), delta_tensor)


def inject_conv2d(inputs: tf.Tensor, 
                  weights: tf.Tensor, 
                  strides: List[int], 
                  padding: str, 
                  inj_args: InjArgs) -> tf.Tensor:
    """
    Perform convolution with fault injection support.
    
    Args:
        inputs: Input tensor
        weights: Weight tensor
        strides: Convolution strides
        padding: Padding type
        inj_args: Injection arguments
        
    Returns:
        Convolution output with injection applied
    """
    # Apply input injection if needed
    if is_input_target(inj_args.inj_type):
        inputs = apply_injection_to_tensor(inputs, inj_args)
    
    # Apply weight injection if needed
    if is_weight_target(inj_args.inj_type):
        weights = apply_weight_injection(weights, inj_args)
    
    # Perform convolution
    conv_output = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding)
    
    # Apply output injection if needed
    if is_output_target(inj_args.inj_type):
        conv_output = apply_injection_to_tensor(conv_output, inj_args)
    
    return conv_output


class InjectableConv2D(tf.keras.layers.Conv2D):
    """
    Convolution layer with fault injection support for TensorFlow 2.19.0.
    
    This layer extends the standard Conv2D layer to support fault injection
    into inputs, weights, or outputs during forward pass.
    """
    
    def __init__(self, 
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = 1,
                 padding: str = 'valid',
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_normal',
                 seed: Optional[int] = None,
                 layer_name: Optional[str] = None,
                 **kwargs):
        
        # Initialize without bias first, we'll add it separately if needed
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,  # We handle bias separately
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed) if seed else kernel_initializer,
            **kwargs
        )
        
        self.layer_name = layer_name or self.name
        self.has_bias = use_bias
        self.seed = seed
        
        if self.has_bias:
            self.bias_layer = BiasLayer(name=f'{self.layer_name}_bias')
    
    def call(self, 
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass with optional fault injection.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            inject: Whether to apply injection
            inj_args: Injection arguments
            
        Returns:
            Tuple of (final_output, conv_output)
        """
        is_target = (inj_args and inj_args.inj_layer == self.layer_name)
        
        if not is_target or not inject:
            # Normal forward pass
            conv_output = super().call(inputs)
        else:
            # Injection-aware forward pass
            def normal_conv():
                return super(InjectableConv2D, self).call(inputs)
            
            def inject_conv():
                if is_input_target(inj_args.inj_type):
                    inputs_inj = apply_injection_to_tensor(inputs, inj_args)
                else:
                    inputs_inj = inputs
                
                if is_weight_target(inj_args.inj_type):
                    # Temporarily modify weights
                    original_weights = self.kernel.numpy()
                    injected_weights = apply_injection_to_tensor(self.kernel, inj_args)
                    self.kernel.assign(injected_weights)
                    
                    conv_out = super(InjectableConv2D, self).call(inputs_inj)
                    
                    # Restore original weights
                    self.kernel.assign(original_weights)
                else:
                    conv_out = super(InjectableConv2D, self).call(inputs_inj)
                
                if is_output_target(inj_args.inj_type):
                    conv_out = apply_injection_to_tensor(conv_out, inj_args)
                
                return conv_out
            
            conv_output = tf.cond(
                tf.reduce_all(inject), 
                inject_conv, 
                normal_conv
            )
        
        # Apply bias if needed
        if self.has_bias:
            final_output = self.bias_layer(conv_output)
        else:
            final_output = conv_output
        
        return final_output, conv_output


class InjectableBatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Batch normalization layer with fault injection support for TensorFlow 2.19.0.
    
    Updated for TensorFlow 2.19.0 with proper momentum handling.
    """
    
    def __init__(self, 
                 layer_name: Optional[str] = None,
                 momentum: float = 0.99,  # Updated default for TF 2.19.0
                 **kwargs):
        super().__init__(momentum=momentum, **kwargs)
        self.layer_name = layer_name or self.name
    
    def call(self, 
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> tf.Tensor:
        """
        Forward pass with optional fault injection.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            inject: Whether to apply injection
            inj_args: Injection arguments
            
        Returns:
            Normalized tensor
        """
        # For now, pass through to parent - injection can be added later
        return super().call(inputs, training=training)


class InjectableReLU(tf.keras.layers.ReLU):
    """
    ReLU activation layer with fault injection support.
    """
    
    def __init__(self, layer_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.layer_name = layer_name or self.name
    
    def call(self, 
             inputs: tf.Tensor,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> tf.Tensor:
        """
        Forward pass with optional fault injection.
        
        Args:
            inputs: Input tensor
            inject: Whether to apply injection
            inj_args: Injection arguments
            
        Returns:
            Activated tensor
        """
        # Apply injection if this layer is targeted
        if inject and inj_args and inj_args.inj_layer == self.layer_name:
            if is_input_target(inj_args.inj_type):
                inputs = apply_injection_to_tensor(inputs, inj_args)
        
        output = super().call(inputs)
        
        # Apply output injection if needed
        if inject and inj_args and inj_args.inj_layer == self.layer_name:
            if is_output_target(inj_args.inj_type):
                output = apply_injection_to_tensor(output, inj_args)
        
        return output


class InjectableDense(tf.keras.layers.Dense):
    """
    Dense layer with fault injection support for TensorFlow 2.19.0.
    """
    
    def __init__(self, 
                 units: int,
                 use_bias: bool = True,
                 activation: Optional[str] = None,
                 kernel_initializer: str = 'glorot_normal',
                 seed: Optional[int] = None,
                 layer_name: Optional[str] = None,
                 **kwargs):
        
        super().__init__(
            units=units,
            use_bias=False,  # Handle bias separately
            activation=activation,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed) if seed else kernel_initializer,
            **kwargs
        )
        
        self.layer_name = layer_name or self.name
        self.has_bias = use_bias
        self.seed = seed
        
        if self.has_bias:
            self.bias_layer = BiasLayer(name=f'{self.layer_name}_bias')
    
    def call(self, 
             inputs: tf.Tensor,
             inject: bool = False,
             inj_args: Optional[InjArgs] = None) -> tf.Tensor:
        """
        Forward pass with optional fault injection.
        
        Args:
            inputs: Input tensor
            inject: Whether to apply injection
            inj_args: Injection arguments
            
        Returns:
            Dense layer output
        """
        is_target = (inj_args and inj_args.inj_layer == self.layer_name)
        
        if not is_target or not inject:
            # Normal forward pass
            dense_output = super().call(inputs)
        else:
            # Apply input injection if needed
            if is_input_target(inj_args.inj_type):
                inputs = apply_injection_to_tensor(inputs, inj_args)
            
            # Apply weight injection if needed
            if is_weight_target(inj_args.inj_type):
                # Temporarily modify weights
                original_weights = [w.numpy() for w in self.weights]
                injected_kernel = apply_injection_to_tensor(self.kernel, inj_args)
                self.kernel.assign(injected_kernel)
                
                dense_output = super().call(inputs)
                
                # Restore original weights
                self.kernel.assign(original_weights[0])
                if len(original_weights) > 1:
                    self.bias.assign(original_weights[1])
            else:
                dense_output = super().call(inputs)
            
            # Apply output injection if needed
            if is_output_target(inj_args.inj_type):
                dense_output = apply_injection_to_tensor(dense_output, inj_args)
        
        # Apply bias if needed
        if self.has_bias:
            final_output = self.bias_layer(dense_output)
        else:
            final_output = dense_output
        
        return final_output


class BiasLayer(tf.keras.layers.Layer):
    """
    Standalone bias layer for TensorFlow 2.19.0.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape: tf.TensorShape):
        """Build the bias weight."""
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add bias to inputs."""
        return inputs + self.bias


# Utility functions for tensor manipulation (updated for TF 2.19.0)
def rotate_180(tensor: tf.Tensor) -> tf.Tensor:
    """
    Rotate tensor by 180 degrees (for convolution backward pass).
    
    Args:
        tensor: Input tensor
        
    Returns:
        Rotated tensor
    """
    return tf.reverse(tensor, axis=[0, 1])


def pad_for_full_conv2d(x: tf.Tensor, 
                       kernel_size: int, 
                       padding: str, 
                       is_input: bool = True) -> tf.Tensor:
    """
    Pad tensor for full convolution operation.
    
    Args:
        x: Input tensor
        kernel_size: Kernel size
        padding: Padding type
        is_input: Whether this is input or weight tensor
        
    Returns:
        Padded tensor
    """
    if padding == "VALID":
        if not is_input:
            return x
        else:
            pad_size = (kernel_size - 1) // 2 * 2
            return tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    else:
        pad_size = (kernel_size - 1) // 2
        return tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])


def convert_NHWC_to_HWIO(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert tensor from NHWC to HWIO format.
    
    Args:
        tensor: Input tensor in NHWC format
        
    Returns:
        Tensor in HWIO format
    """
    return tf.transpose(tensor, perm=[1, 2, 0, 3])


def convert_HWIO_to_NHWC(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert tensor from HWIO to NHWC format.
    
    Args:
        tensor: Input tensor in HWIO format
        
    Returns:
        Tensor in NHWC format
    """
    return tf.transpose(tensor, perm=[2, 0, 1, 3])


# Legacy compatibility aliases
InjectConv2D = InjectableConv2D
InjectBatchNormalization = InjectableBatchNormalization
InjectReLU = InjectableReLU
InjectDense = InjectableDense
inj_to_tensor = apply_injection_to_tensor
inj_to_tensor_wt_tensor = apply_weight_injection
inject_nn_conv2d = inject_conv2d
tf_rot180 = rotate_180
tf_pad_to_full_conv2d = pad_for_full_conv2d
tf_NHWC_to_HWIO = convert_NHWC_to_HWIO
tf_HWIO_to_NHWC = convert_HWIO_to_NHWC
