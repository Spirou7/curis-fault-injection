from enum import Enum

class InjType(Enum):
    INPUT = 1
    INPUT_16 = 2
    WT = 3
    WT_16 = 4
    RBFLIP = 5
    RD = 6
    RD_CORRECT = 7
    ZERO = 8
    N16_RD = 9
    N16_RD_CORRECT = 10
    RD_GLB = 11
    RD_CORRECT_GLB = 12
    N64_INPUT = 13
    N64_WT = 14
    N64_INPUT_16 = 15
    N64_WT_16 = 16
    N64_INPUT_GLB = 17
    N64_WT_GLB = 18

def is_bflip(inj_type):
    return inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16, InjType.RBFLIP]

def is_random(inj_type):
    return inj_type in [InjType.RD, InjType.N16_RD, InjType.RD_GLB]

def is_zero(inj_type):
    return inj_type in [InjType.ZERO]

def is_correct(inj_type):
    return inj_type in [InjType.RD_CORRECT, InjType.N16_RD_CORRECT, InjType.RD_CORRECT_GLB, InjType.N64_INPUT, InjType.N64_WT, InjType.N64_INPUT_16, InjType.N64_WT_16, InjType.N64_INPUT_GLB, InjType.N64_WT_GLB]

def is_input_target(inj_type):
    return inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.N64_INPUT, InjType.N64_INPUT_16, InjType.N64_INPUT_GLB]

def is_weight_target(inj_type):
    return inj_type in [InjType.WT, InjType.WT_16, InjType.N64_WT, InjType.N64_WT_16, InjType.N64_WT_GLB]

def is_output_target(inj_type):
    return inj_type in [InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO, InjType.N16_RD, InjType.N16_RD_CORRECT, InjType.RD_GLB, InjType.RD_CORRECT_GLB]

def num_inj(
    inj_type: InjType
) -> tuple[int, int]:
    """
    Returns the number of injection positions and the number of consecutive injections to place out of these options.
    Args:
        inj_type (InjType): The injection type.
    Returns:
        n_inj (int): The number of injection positions.
        n_repeat (int): The number of consecutive injections to place out of these options.
    """

    if inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16, InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO]:
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
        raise ValueError(f"Invalid injection type: {inj_type}")

class InjState(Enum):
    IDLE = 1
    GOLDEN = 2
    INJECT = 3