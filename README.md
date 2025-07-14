pip install tensorflow-tpu==2.18.0 --find-links=https://storage.googleapis.com/libtpu-tf-releases/index.html
pip install tensorflow=2.18.0


# CURIS Fault Injection Framework - TensorFlow 2.19.0

A modernized fault injection framework for studying hardware fault tolerance in deep neural network training, updated for TensorFlow 2.19.0.

## Overview

This framework enables systematic fault injection experiments in deep learning training to study:
- Hardware fault tolerance characteristics
- Training robustness under various fault models
- Fault propagation patterns in neural networks
- Recovery mechanisms and fault masking effects

## Key Features

### TensorFlow 2.19.0 Modernization
- **Updated APIs**: Modern TensorFlow 2.19.0 APIs replacing deprecated functions
- **Mixed Precision**: Support for mixed precision training 
- **XLA Compilation**: Optional XLA compilation for performance
- **Distributed Training**: GPU/TPU distributed training support
- **Type Safety**: Comprehensive type hints throughout codebase

### Fault Injection Capabilities
- **18+ Fault Types**: Bit flips, random values, zero injection, correct value injection
- **Multiple Targets**: Input activations, weights, output activations
- **Injection Stages**: Forward pass and backward pass injection
- **Layer Targeting**: Precise layer-level fault targeting
- **Reproducible Experiments**: Deterministic fault injection for research

### Supported Models
- ResNet-18 (fully implemented)
- EfficientNet (architecture ready)
- DenseNet (architecture ready)
- NFNet (architecture ready)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd curis-fault-injection

# Install dependencies
pip install tensorflow==2.19.0 numpy pandas

# Verify installation
python test_migration.py
```

## Quick Start

### Basic Training without Injection
```python
from models.resnet_18 import create_resnet18
from tools.utils import load_cifar10_data, setup_tensorflow_config

# Setup TensorFlow
setup_tensorflow_config()

# Load data
train_dataset, test_dataset, _, _ = load_cifar10_data(seed=123)

# Create model
model = create_resnet18(seed=123)

# Train normally
model.fit(train_dataset.batch(32), epochs=10)
```

### Fault Injection Experiment
```python
# Create injection configuration
python main.py --config experiments/resnet18_rd_injection.csv --output_dir results/
```

## Architecture

```
curis-fault-injection/
├── injection/                 # Core fault injection system
│   ├── injection_types.py     # Fault type definitions
│   ├── injection_args.py      # Parameter handling
│   └── injection_utils.py     # Core injection utilities
├── models/                    # Neural network models
│   └── resnet_18.py          # ResNet-18 implementation
├── building_blocks/           # Reusable components
│   ├── residual_block.py     # ResNet building blocks
│   └── inject_layers.py      # Injection-aware layers
├── tools/                     # Utilities and helpers
│   └── utils.py              # Data loading, TF config
├── config.py                 # Configuration parameters
├── main.py                   # Training orchestration
└── test_migration.py         # Integration tests
```

## Configuration Format

Injection experiments are configured via CSV files:

```csv
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
```

### Configuration Parameters
- `model`: Model architecture (resnet18, effnet, densenet, nfnet)
- `stage`: Injection stage (fwrd_inject, bkwd_inject)
- `fmodel`: Fault model (RD, RBFLIP, ZERO, etc.)
- `target_layer`: Specific layer to target
- `target_epoch`/`target_step`: When to inject
- `inj_pos`: Injection positions (tensor coordinates)
- `inj_values`: Fault values to inject

## Fault Types

The framework supports 18+ fault injection types:

### Basic Faults
- `INPUT`: Single input activation fault
- `WT`: Single weight fault  
- `RBFLIP`: Random bit flip
- `RD`: Random value injection
- `ZERO`: Zero injection

### Multi-Position Faults
- `N16_RD`: 16-position random injection
- `N64_INPUT`: 64-position input injection
- `RD_GLB`: Global random injection (1000 repetitions)

### Correct Value Injection
- `RD_CORRECT`: Inject correct values from other positions
- `N64_INPUT_GLB`: Global 64-position input injection

## API Reference

### Core Classes

#### `InjType` (Enum)
Defines all supported fault injection types.

#### `SimulationParameters` (Dataclass)
Container for experiment configuration parameters.

#### `InjectionArgs` (Dataclass)
Runtime injection arguments for fault execution.

### Key Functions

#### `read_injection_config(file_path: str) -> SimulationParameters`
Parse CSV configuration file into parameters.

#### `choose_inj_pos(target, inj_type, inj_args) -> Tuple[mask, delta]`
Select injection positions and generate fault values.

#### `create_resnet18(seed: int = 123) -> ResNet18`
Create ResNet-18 model with injection support.

## Migration from TensorFlow 2.6.0

This codebase has been systematically migrated from TensorFlow 2.6.0 to 2.19.0:

### API Updates
- `tf.keras.layers.RandomFlip` (was `experimental.preprocessing`)
- `tf.keras.utils.set_random_seed()` (modern seed management)
- `tf.keras.optimizers` (updated optimizer APIs)
- Batch normalization momentum handling (inverted meaning)

### Architecture Improvements
- Modular design with clear separation of concerns
- Type-safe interfaces with comprehensive type hints
- Modern Python patterns (dataclasses, type unions)
- Improved error handling and validation

### Backward Compatibility
- Legacy function aliases maintained (`is_bflip`, `bin2fp32`, etc.)
- Original experiment configurations supported
- Same fault injection semantics preserved

## Testing

Run the integration test suite:

```bash
python test_migration.py
```

Tests verify:
- ✅ Injection type system functionality
- ✅ Binary conversion utilities
- ✅ Configuration file parsing
- ✅ Model creation and forward pass
- ✅ Residual block functionality
- ✅ Injectable layer operations
- ✅ Data loading pipeline
- ✅ TensorFlow 2.19.0 compatibility

## Performance Optimizations

### TensorFlow 2.19.0 Features
```python
from tools.utils import setup_tensorflow_config

# Enable performance optimizations
setup_tensorflow_config(
    mixed_precision=True,    # 16-bit training
    xla_compile=True,        # XLA acceleration
    memory_growth=True       # Dynamic GPU memory
)
```

### Distributed Training
```python
# Automatic GPU/TPU detection
python main.py --config experiment.csv --use_tpu
```

## Research Applications

### Fault Tolerance Studies
- Characterize training robustness under hardware faults
- Study fault propagation patterns in deep networks
- Evaluate recovery mechanisms and fault masking

### Hardware Reliability
- Model hardware failure scenarios
- Test error detection and correction mechanisms
- Validate fault-tolerant training algorithms

### Systematic Evaluation
- Reproducible fault injection experiments
- Statistical analysis across multiple fault types
- Comparative studies across different architectures

## Contributing

1. Follow the existing code style and type hints
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure backward compatibility when possible

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{curis_fault_injection,
  title={Fault Injection Framework for Deep Learning Training Robustness},
  author={[Original Authors]},
  booktitle={[Conference]},
  year={[Year]}
}
```

## License

[License information]

---

**Note**: This is a modernized version of the original fault injection framework, updated for TensorFlow 2.19.0 while maintaining full backward compatibility with existing experiments and research workflows.
