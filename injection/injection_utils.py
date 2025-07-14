"""
Core injection utilities for TensorFlow 2.19.0 fault injection framework

This module provides the core functionality for fault injection including:
- Fault value generation (bit flips, random values, etc.)
- Position selection for injection
- Layer targeting and selection
- Injection execution utilities
"""

import struct
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Union
from io import TextIOWrapper

from injection.injection_types import *
from injection.injection_args import InjArgs, log_injection_info
from injection.simulation_parameters import SimulationParameters


def binary_to_float32(binary_string: str) -> float:
    """Convert 32-bit binary string to IEEE 754 float32."""
    if len(binary_string) != 32:
        raise ValueError(f"Binary string must be 32 bits, got {len(binary_string)}")
    return struct.unpack('!f', struct.pack('!I', int(binary_string, 2)))[0]


def float32_to_binary(value: float) -> str:
    """Convert IEEE 754 float32 to 32-bit binary string."""
    return ''.join(format(c, '08b') for c in struct.pack('!f', value))


# Backward compatibility aliases
bin2fp32 = binary_to_float32
fp322bin = float32_to_binary

def choose_inj_pos(
    target: np.ndarray,
    inj_type: InjType,
    injection_args: InjArgs,
    train_recorder: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly choose injection positions for the target tensor.
    Args:
        target (np.ndarray): The target tensor to inject into.
        inj_type (InjType): The type of injection to perform.
        injection_args (InjArgs): The injection arguments.
    Returns:
        mask (np.ndarray): The mask tensor.
        delta (np.ndarray): The delta tensor.
    """
    np.random.seed(None)
    shape = target.shape


    def random_positions(
        shape: tuple[int, ...],
        n_inj: int,
        n_repeat: int
    ) -> list[tuple[int, ...]]:
        """
        Randomly choose injection positions for the target tensor.
        Args:
            shape (tuple[int, ...]): The shape of the target tensor.
            train_recorder (file): The file to record the injection positions.
            n_inj (int): The number of injection positions to choose.
            n_repeat (int): The number of consecutive injections to place out of these options.
        Returns:
            positions (list[tuple[int, ...]]): The injection positions.
        """
        positions = []
        l = len(shape)

        total = np.prod(shape) / n_inj
        start = np.random.randint(int(total))
        end = np.random.randint(start+1, int(total)) if n_repeat != 1 else start + 1
        start_pos = np.unravel_index(start * n_inj, shape)
        end_pos = np.unravel_index(end * n_inj, shape)
        for flat in range(start * n_inj, end * n_inj):
            positions.append(np.unravel_index(flat, shape))
        return positions

    # get the number of injection positions and the number of consecutive injections to place out of these options.
    n_inj, n_repeat = get_injection_count(inj_type)

    # randomly choose injection positions for the target tensor.
    positions = random_positions(shape, n_inj, n_repeat)


    def flip_one_bit(target: float) -> float:
        """
        Flip one bit of the target float.
        Args:
            target (float): The target float.
        Returns:
            result (float): The flipped float.
        """
        np.random.seed(None)
        bin_target = float32_to_binary(target)
        flip = np.random.randint(32)
        bin_output = ""
        for i in range(32):
            if i == flip:
                bin_output += ('1' if bin_target[i] == '0' else '0')
            else:
                bin_output += bin_target[i]
        return binary_to_float32(bin_output) 

    def get_random_value() -> float:
        """
        Get a random value by randomly selecting a 32-bit binary string and converting it to a float.
        Returns:
            result (float): The random value.
        """
        np.random.seed(None)
        one_bin = ''
        result = 0
        while one_bin == '' or not np.isfinite(result):
            one_bin = ''
            for _ in range(32):
                one_bin += str(np.random.randint(0,2))
            result = struct.unpack('!f',struct.pack('!I', int(one_bin, 2)))[0]
        return result

    def get_random_correct(target: np.ndarray) -> float:
        """
        Get a random correct value by randomly selecting a position in the target tensor and returning the value at that position.
        Args:
            target (np.ndarray): The target tensor.
        Returns:
            result (float): The random correct value.
        """
        shape = target.shape
        rd_pos = np.unravel_index(np.random.randint(np.prod(shape)), shape)
        return target[rd_pos].item()

    # initialize the mask and delta tensors.
    mask = np.ones(shape)
    delta = np.zeros(shape)

    # set the injection positions in the arguments
    injection_args.inj_pos = positions

    # inject data into the target tensor.
    for pos in positions:
        # get the original value at the injection position.
        ori_val = target[pos]

        if is_random_value(inj_type):
            val_delta = get_random_value()
        elif is_bit_flip(inj_type):
            val_delta = flip_one_bit(ori_val)
        elif is_correct_value(inj_type):
            val_delta = get_random_correct(target)
        else:
            val_delta = 0

        # set the mask and delta tensors.
        mask[pos] = 0
        delta[pos] = val_delta

        log_injection_info(train_recorder, "Position is {}, Golden data is {}, inject data is {}\n".format(pos, ori_val, val_delta))
        
        if injection_args.inj_values is None:
            injection_args.inj_values = []
        
        # add this injection delta to the injection arguments.
        injection_args.inj_values.append(val_delta)

    return mask, delta

def get_target_tensor(
    inj_type: InjType,
    inj_replica: int,
    layer_inputs: tf.distribute.DistributedValues,
    layer_kernels: tf.distribute.DistributedValues,
    layer_outputs: tf.distribute.DistributedValues,
) -> np.ndarray:
    """
    Accesses the tensor on a specific replica within a distributed context.

    This function is designed to be run within `strategy.run()`.
    """
    # 1. Select the distributed value based on injection type
    if is_input_target(inj_type):
        dist_value = layer_inputs
    elif is_weight_target(inj_type):
        # Handle cases where kernels might be a list of distributed values
        dist_value = layer_kernels[0] if isinstance(layer_kernels, list) else layer_kernels
    elif is_output_target(inj_type):
        dist_value = layer_outputs
    else:
        tf.print("ERROR: Unsupported inject type!")
        # Returning an empty tensor or raising an error within the graph
        return np.array([])

    # 2. Get the replica context
    replica_context = tf.distribute.get_replica_context()
    # Get the ID of the current replica
    replica_id = replica_context.replica_id_in_sync_group

    # 3. Conditionally access the tensor on the target replica - TPU compatible
    if tf.equal(replica_id, inj_replica):
        # Access the local tensor value for this replica without .numpy() for TPU compatibility
        target_tensor = dist_value.values[replica_id]
    else:
        # Other replicas can return an empty tensor
        target_tensor = tf.constant([])

    return target_tensor






def get_injection_args(
    simulation_parameters: SimulationParameters,
    layer_inputs: tf.distribute.DistributedValues,
    layer_kernels: tf.distribute.DistributedValues,
    layer_outputs: tf.distribute.DistributedValues,
) -> InjArgs:
    """
    Get the injection arguments for the simulation parameters.
    """

    inj_args = InjArgs(
        inj_replica=simulation_parameters.inj_replica,
        inj_layer=simulation_parameters.inj_layer,
        inj_type=simulation_parameters.inj_type,
    )

    np.random.seed(None)
    inj_replica = np.random.randint(simulation_parameters.strategy.num_replicas_in_sync)
    inj_args.inj_replica = inj_replica

    target = get_target_tensor(simulation_parameters.inj_type, simulation_parameters.inj_replica, layer_inputs, layer_kernels, layer_outputs)

    mask, delta = choose_inj_pos(target, simulation_parameters.inj_type, inj_args)

    # TPU-compatible tensor handling - avoid .numpy() calls
    if type(layer_kernels) == list:
        inj_args.golden_weights = []
        for elem in layer_kernels:
            inj_args.golden_weights.append(elem.values[0])
    else:
        inj_args.golden_weights = layer_kernels.values[0]

    inj_args.golden_output = layer_outputs.values[0]

    np_array = np.zeros(simulation_parameters.strategy.num_replicas_in_sync, dtype=bool)
    np_array[inj_replica] = True
    inj_flag_dataset = tf.data.Dataset.from_tensor_slices(np_array).repeat().batch(simulation_parameters.strategy.num_replicas_in_sync)
    inj_flag_iterator = iter(simulation_parameters.strategy.experimental_distribute_dataset(inj_flag_dataset))
    inj_flag = next(inj_flag_iterator)

    return inj_args
