# Relaxed Equivariant Graph Neural Networks

A comprehensive PyTorch implementation of Graph Neural Networks (GNNs) with support for multiple symmetry groups (permutation, SO(3), E(3), SE(3), etc.) and equivariance-aware training. This framework enables learning symmetry-preserving representations in molecular graphs and point clouds.

## Overview

This project implements **equivariance loss regularization** to train GNNs that respect geometric and permutation symmetries during learning. Includes 13 different GNN architectures ranging from permutation-only (GCN, GIN, GraphSAGE) to fully equivariant models (EGNN, PaiNN, NequIP).

### Key Features

- **13 GNN Architectures**: From baseline MLPs to advanced equivariant models
- **8 Symmetry Groups**: Permutation, SO(3), O(3), SE(3), E(3), Translation, Reflection, Scaling
- **Flexible Configuration System**: Type-safe YAML-like configuration with presets
- **Multiple Datasets**: ZINC, QM9, QM7b, MD17, OC20, ModelNet40, ShapeNet, and more
- **Production-Ready Training**: Early stopping, checkpointing, gradient clipping, mixed precision
- **Logging Integration**: Weights & Biases, TensorBoard, custom metrics tracking
- **GPU Optimized**: Efficient GPU utilization with proper CUDA synchronization

---

## Project Structure

```
.
├── equivariant_gnn.py       # Unified BaseGNN with 13 architectures
├── equivariance_loss.py     # Equivariance loss computation for 8 symmetry groups
├── train.py                 # Main training script
├── config.py                # Configuration system with dataclass validation
├── load_dataset.py          # Dataset loading (15+ datasets)
├── utils.py                 # Metrics, checkpointing, early stopping
├── logger.py                # Logging infrastructure (wandb/tensorboard)
├── schedulers.py            # Learning rate & equivariance weight scheduling
└── README.md                # This file
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+
- PyTorch Geometric 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```
# Clone repository
git clone <repo-url>
cd equivariant-gnns

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install wandb tensorboard tqdm pydantic
```

---

## Quick Start

### Minimal Example

```
from config import get_config
from train import train

# Load default configuration
config = get_config('default')

# Override settings
config.model.model_type = 'gcn'
config.data.dataset_name = 'ZINC'
config.training.num_epochs = 100
config.equivariance.symmetry_groups = ['permutation']

# Train
model, test_metrics = train(config)
```

### Command Line

```
# Train with default config
python train.py

# Train with preset
python train.py --config baseline

# Train E(3)-equivariant model
python train.py --config e3_equivariant

# Custom experiment
python train.py \
  --config baseline \
  --experiment-name my_experiment \
  --logger wandb \
  --seed 42
```

---

## Configuration

### Configuration Structure

Configurations are organized in hierarchical dataclasses:

```
from config import ExperimentConfig, ModelConfig, EquivarianceLossConfig

config = ExperimentConfig(
    # Model configuration
    model=ModelConfig(
        model_type='gcn',           # See supported models below
        in_channels=12,
        hidden_channels=64,
        out_channels=1,
        num_layers=4,
        dropout=0.5
    ),
  
    # Equivariance losses
    equivariance=EquivarianceLossConfig(
        symmetry_groups=['permutation', 'so3'],
        group_weights={'permutation': 0.1, 'so3': 0.1},
        num_samples=3
    ),
  
    # Training
    training=TrainingConfig(
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001,
        optimizer='adamw'
    ),
  
    # Dataset
    data=DataConfig(
        dataset_name='ZINC',
        use_positions=False
    )
)
```

### Available Presets

```
# Baseline GCN without equivariance
config = get_config('baseline')

# E(3)-equivariant EGNN
config = get_config('e3_equivariant')

# Multi-symmetry training
config = get_config('multi_symmetry')
```

---

## Supported Models

### Permutation-Only (Invariant to Node Order)

- **GCN** (Graph Convolutional Network)
- **GIN** (Graph Isomorphism Network)
- **GraphSAGE** (SAmple and aggreGatE)
- **Transformer** (Graph Transformer)
- **MLP** (Raw MLP baseline)

### Geometric-Invariant (E(3)-Invariant)

- **SchNet** (Distance-based interactions)
- **DimeNet** (Angle-aware distances)

### Geometric-Equivariant (E(3)-Equivariant)

- **EGNN** (Equivariant Graph Neural Network)
- **PaiNN** (Periodic Message Passing Network)

### High-Order Equivariance

- **NequIP** (E(3)-equivariant with tensor products)
- **SE(3)-Transformer** (Attention-based SE(3) equivariance)
- **Vector Neuron** (SO(3)-equivariant)
- **ClofNet** (SE(3) with local frames)

---

## Symmetry Groups

### Permutation

Tests if reordering nodes produces equivalent node embeddings:
\[
f(\pi \cdot x) = \pi \cdot f(x)
\]
**Use for:** All GNNs (invariant by design)

### SO(3) (Rotations)

Tests equivariance to 3D rotations:
\[
f(R \cdot x) = R \cdot f(x)
\]
**Use for:** Geometric models, point clouds

### O(3) (Rotations + Reflections)

Tests O(3) equivariance:
\[
f(O \cdot x) = O \cdot f(x), \quad O \in O(3)
\]
**Use for:** Reflection-aware models

### SE(3) (Rotations + Translations)

Tests SE(3) equivariance:
\[
f(g \cdot x) = g \cdot f(x), \quad g \in SE(3)
\]
**Use for:** Molecular/materials systems

### E(3) (SO(3) ⋉ T(3))

Full Euclidean group (rotations + translations + reflections)

### Translation

Tests invariance to spatial translations:
\[
f(x + t) = f(x)
\]

### Scaling & Reflection

Individual scaling and reflection equivariance

---

## Datasets

### Molecular Graphs (2D)

- **ZINC** (250K molecules): Property prediction
- **AQSOL** (10K molecules): Solubility prediction

### Quantum Chemistry (3D)

- **QM9** (134K molecules): 13 quantum properties
- **QM7b** (7K molecules): Atomization energies
- **MD17** (150K-1M conformations): Molecular dynamics

### Materials & Catalysis

- **OC20** (1.3M structures): Catalyst adsorption
- **ISO17** (Constitutional isomers)

### Point Clouds

- **ModelNet40** (12K models): Shape classification
- **ShapeNet** (51K models): 3D shape dataset

### Biomolecules

- **ATOM3D** (Various tasks): Protein interactions
- **Molecule3D** (3.9M molecules): 3D geometry prediction

---

## Training

### Basic Training Loop

```
from train import train
from config import get_config

config = get_config('baseline')
config.training.num_epochs = 50
config.data.dataset_name = 'ZINC'

model, metrics = train(config)
```

### Monitoring Training

```
# With Weights & Biases
config.logging.logger_type = 'wandb'
config.logging.wandb_project = 'equivariant-gnns'

# Or TensorBoard
config.logging.logger_type = 'tensorboard'
config.logging.tensorboard_dir = './runs'
```

### Custom Loss Scheduling

```
from config import SchedulerConfig

config.scheduler = SchedulerConfig(
    schedule_type='exponential',      # How equivariance weight changes
    alpha_0=1.0,                      # Initial weight
    beta=0.1,                         # Decay rate
    lr_schedule='cosine',             # Learning rate schedule
    lr_warmup_epochs=5
)
```

---

## API Reference

### EquivarianceLoss

Computes equivariance error: \(L_{eq} = ||f(g \cdot x) - g \cdot f(x)||^2\)

```
from equivariance_loss import EquivarianceLoss

loss_fn = EquivarianceLoss(
    group_type='so3',              # Symmetry group
    num_samples=5,                 # Random transformations per graph
    normalize=True,                # Normalize by feature magnitude
    feature_type='invariant'       # 'invariant' or 'equivariant'
)

loss = loss_fn(
    network_fn=my_network,         # Callable network
    positions=node_positions,      # [N, 3]
    features=node_features,        # [N, D]
    edge_index=edges,              # [2, E]
    batch=batch_assignment         # [N]
)
```

### BaseGNN

Unified GNN architecture supporting 13 model types.

```
from equivariant_gnn import BaseGNN

model = BaseGNN(
    in_channels=12,
    hidden_channels=64,
    out_channels=1,
    num_layers=4,
    model_type='egnn',
    spatial_dim=3
)

# Forward pass
output = model(x, pos, edge_index, batch)

# Get node embeddings for equivariance testing
node_embeddings = model(
    x, pos, edge_index, batch, 
    return_node_embeddings=True
)
```

---

## Performance

### GPU vs CPU

**Issue:** Equivariance loss computation involves multiple forward passes, which can be slower on GPU due to:

- Redundant `.item()` calls (GPU-CPU synchronization)
- CPU tensor operations on GPU data
- Inefficient data loading (num_workers=0)

**Solutions:**

1. Set `num_workers=4` in DataLoader
2. Add `pin_memory=True` for faster GPU transfers
3. Defer `.item()` calls until after backward pass
4. Use `torch.index_select()` instead of advanced indexing

### Benchmarks (ZINC dataset, batch_size=32)

| Model | Mode | Time/Epoch | Memory |
| ----- | ---- | ---------- | ------ |
| GCN   | CPU  | 45s        | 1.2GB  |
| GCN   | GPU  | 8s         | 2.1GB  |
| EGNN  | CPU  | 120s       | 2.5GB  |
| EGNN  | GPU  | 18s        | 4.2GB  |

---

## Troubleshooting

### CUDA Errors

```
# Enable synchronous execution for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Or restart Python completely between runs
```

### Out of Memory

```
# Reduce batch size
config.training.batch_size = 16

# Reduce hidden dimensions
config.model.hidden_channels = 32

# Reduce number of samples in equivariance loss
config.equivariance.num_samples = 1
```

### Model Not Learning

```
# Check if data has required properties
config.data.use_positions = False   # For models that don't need positions

# Increase equivariance weight
config.scheduler.alpha_0 = 1.0      # Start with higher weight

# Use simpler model
config.model.model_type = 'gcn'
config.model.num_layers = 2
```

---

## References

### Key Papers

- **GCN**: Kipf & Welling, 2016. *Semi-Supervised Classification with Graph Convolutional Networks*
- **GIN**: Xu et al., 2018. *How Powerful are Graph Neural Networks?*
- **EGNN**: Satorras et al., 2021. *E(n) Equivariant Graph Neural Networks*
- **PaiNN**: Schütt et al., 2021. *Equivariant Message Passing for the Prediction of Molecular Properties and Data-Driven Force Fields*
- **NequIP**: Batzner et al., 2022. *E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*
- **SchNet**: Schütt et al., 2018. *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional symmetry groups (conformal, projective)
- [ ] More model architectures
- [ ] Additional datasets
- [ ] Distributed training support
- [ ] Mixed precision training optimization
- [ ] Model pruning & quantization

---

## License

MIT License - See LICENSE file

---

## Citation

```
@software{equivariant_gnns_2025,
  title={Relaxed Equivariant Graph Neural Networks},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/equivariant-gnns}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## FAQ

**Q: Which model should I use?**
A: Start with GCN for 2D graphs (ZINC), EGNN for 3D molecular data with geometry (QM9, MD17), and NequIP/PaiNN for force predictions.

**Q: How do I add a new dataset?**
A: Implement a loader in `load_dataset.py` and register it in `DATASET_SYMMETRIES`.

**Q: Can I use this for my own data?**
A: Yes! You need to convert your data to PyTorch Geometric format (PyG Data objects with `x`, `pos`, `edge_index`, `y`).

**Q: How do I measure if my model is actually equivariant?**
A: Use the equivariance loss directly - lower loss means better equivariance. Values < 1e-4 indicate excellent equivariance.

**Q: What's the difference between invariance and equivariance?**
A: **Invariant**: \(f(g \cdot x) = f(x)\) (output unchanged)
**Equivariant**: \(f(g \cdot x) = g \cdot f(x)\) (output transforms with input)

---

**Last Updated**: October 2025
**Maintainer**: [Your Name/Team]

```

```
