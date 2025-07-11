import numpy as np
from injection.injection_types import *
from injection.random_injection_layers import choose_random_layer
from config import *

class SimulationParameters:
    def __init__(self,
        model: str,
        phase: str,
        inj_replica: int,
        inj_layer: str,
        inj_type: InjType,
        strategy: object,
        target_epoch: int,
        target_step: int,
        learning_rate: float,
        seed: int,
    ):
        self.inj_replica = inj_replica
        self.inj_layer = inj_layer
        self.inj_type = inj_type

def decide_random_simulation_parameters(
    model: str | None = None,
    phase: str | None = None,
    layer: str | None = None,
    fault_model: InjType | None = None,
    strategy: object | None = None,
    target_worker: int | None = None,
    target_epoch: int | None = None,
    target_step: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None
) -> SimulationParameters:
    """
    Create a set of random simulation parameters, unless you specify a specific parameter to constrain.
    """

    # ensure a model is selected
    if model is None:
        models = ['resnet']
        model = models[np.random.randint(len(models))]
    
    # ensure a phase is selected
    if phase is None:
        phases = ['fwrd', 'bkwd']
        phase = phases[np.random.randint(len(phases))]
    
    # ensure a layer is selected
    if layer is None:
        layer = choose_random_layer(model, phase)
    
    if(fault_model is None):
        f_models = list(InjType)
        fault_model = f_models[np.random.randint(len(f_models))]
    
    if(target_worker is None):
        target_worker = np.random.randint(strategy.num_replicas_in_sync)
    
    if(target_epoch is None):
        target_epoch = np.random.randint(EPOCHS)
    
    if(target_step is None):
        target_step = np.random.randint(10)
    
    if(learning_rate is None):
        learning_rate = np.random.uniform(0.0001, 0.01)
    
    if(seed is None):
        seed = np.random.randint(1000000)

    return SimulationParameters(
        model=model,
        phase=phase,
        inj_replica=target_worker,
        inj_layer=layer,
        inj_type=fault_model,
        strategy=strategy,
        target_epoch=target_epoch,
        target_step=target_step,
        learning_rate=learning_rate,
        seed=seed,
    )

