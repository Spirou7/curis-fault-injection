from injection.injection_types import InjType

class InjArgs():
    def __init__(self, inj_replica, inj_layer, inj_type, golden_weights, golden_output, mask=None, delta=None, inj_pos=None, inj_values=None):
        if not isinstance(inj_type, InjType):
            print("ERROR: Invalid injection type!")
            exit(12)
        self.inj_replica = inj_replica
        self.inj_layer = inj_layer
        self.inj_type = inj_type
        self.golden_weights = golden_weights
        self.golden_output = golden_output

        # Two numpy arrays
        self.inj_mask = mask
        self.inj_delta = delta
        self.inj_pos = inj_pos
        self.inj_values = inj_values