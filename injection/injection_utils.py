# import packages
from enum import Enum
from io import TextIOWrapper
import tensorflow as tf
import numpy as np
import struct
from injection.injection_types import *
from injection.injection_args import InjArgs
from tools.utils import record, fp322bin, bin2fp32

def choose_inj_pos(
    target: np.ndarray,
    inj_type: InjType,
    train_recorder: TextIOWrapper,
    injection_args: InjArgs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly choose injection positions for the target tensor.
    Args:
        target (np.ndarray): The target tensor to inject into.
        inj_type (InjType): The type of injection to perform.
        train_recorder (file): The file to record the injection positions.
        injection_args (InjArgs): The injection arguments.
    Returns:
        mask (np.ndarray): The mask tensor.
        delta (np.ndarray): The delta tensor.
    """
    np.random.seed(None)
    shape = target.shape


    def random_positions(
        shape: tuple[int, ...],
        train_recorder: TextIOWrapper,
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
    n_inj, n_repeat = num_inj(inj_type)

    # randomly choose injection positions for the target tensor.
    positions = random_positions(shape, train_recorder, n_inj, n_repeat)


    def flip_one_bit(target: float) -> float:
        """
        Flip one bit of the target float.
        Args:
            target (float): The target float.
        Returns:
            result (float): The flipped float.
        """
        np.random.seed(None)
        bin_target = fp322bin(target)
        flip = np.random.randint(32)
        bin_output = ""
        for i in range(32):
            if i == flip:
                bin_output += ('1' if bin_target[i] == '0' else '0')
            else:
                bin_output += bin_target[i]
        return bin2fp32(bin_output) 

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

        if is_random(inj_type):
            val_delta = get_random_value()
        elif is_bflip(inj_type):
            val_delta = flip_one_bit(ori_val)
        elif is_correct(inj_type):
            val_delta = get_random_correct(target)
        else:
            val_delta = 0

        # set the mask and delta tensors.
        mask[pos] = 0
        delta[pos] = val_delta

        record(train_recorder, "Position is {}, Golden data is {}, inject data is {}\n".format(pos, ori_val, val_delta))
        
        if injection_args.inj_values is None:
            injection_args.inj_values = []
        
        # add this injection delta to the injection arguments.
        injection_args.inj_values.append(val_delta)

    return mask, delta
