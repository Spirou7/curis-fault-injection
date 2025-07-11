from injection.injection_types import InjType
import numpy as np

class InjArgs():
    def __init__(
        self,
        inj_replica: int,
        inj_layer: int,
        inj_type: InjType,
        golden_weights: np.ndarray,
        golden_output: np.ndarray,
        mask: np.ndarray | None = None,
        delta: np.ndarray | None = None,
        inj_pos: list[tuple[int, ...]] | None = None,
        inj_values: list[float] | None = None,
    ):
        """
        Initialize the injection arguments.
        Args:
            inj_replica (int): The injection replica (needs to be decided by the tpu strategy).
            inj_layer (int): The injection layer.
            inj_type (InjType): The injection type.
            golden_weights (np.ndarray): The golden weights.
            golden_output (np.ndarray): The golden output.
            mask (np.ndarray): The mask.
            delta (np.ndarray): The delta.
            inj_pos (list[tuple[int, ...]]): The injection positions.
            inj_values (list[float]): The injection values.
        """
        if not isinstance(inj_type, InjType):
            print("ERROR: Invalid injection type!")
            exit(12)
        self.inj_replica = inj_replica
        self.inj_layer = inj_layer
        self.inj_type = inj_type
        self.golden_weights = golden_weights
        self.golden_output = golden_output
        self.inj_mask = mask
        self.inj_delta = delta
        self.inj_pos = inj_pos
        self.inj_values = inj_values