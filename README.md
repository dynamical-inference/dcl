# Self-supervised contrastive learning performs non-linear system identification

[![Website](https://img.shields.io/badge/%F0%9F%94%97_Website-blue)](https://dynamical-inference.ai/dcl)
[![Paper](https://img.shields.io/badge/📑_Paper-arXiv-red)](https://arxiv.org/abs/2410.14673)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2025-green)](https://openreview.net/forum?id=ONfWFluZBI)

This repo contains the code to run Dynamics Contrastive Learning (DCL) presented in "Self-supervised contrastive learning performs non-linear system identification".

DCL is a self-supervised learning algorithm that performs non-linear system identification in latent space. It combines contrastive learning with dynamics modeling to learn representations of time-series data.

![Overview figure of dynamics contrastive learning](https://dynamical-inference.ai/dcl/overview.svg)

## 🚀 Quick Navigation

- [Getting Started](#-getting-started)
- [How to use DCL](#-how-to-use-dcl)
- [Examples](#-examples)
- [Reproducing the results](#-reproducing-the-results)
- [Reference](#-reference)

## 🛠 Getting Started

Set up your environment with these steps:

```bash
# Create and activate environment
conda create --name dcl python=3.10
conda activate dcl

# Install Torch according to your setup (see https://pytorch.org/).
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
pip install ./deps/*

# Install the dcl package
pip install -e .
```

## 🔍 How to use DCL

DCL provides a flexible framework for learning dynamics in latent space. Here's a basic example:

```python
from dcl.datasets.timeseries import TensorDataset
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.encoder import MLP
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.solver.optimizer import DCLAdamOptimizer
from dcl.criterions.contrastive import MseInfoNCE
from dcl.models.dynamics.slds import MSESwitchingModel
import torch

# 1. Create or load your dataset
dataset = TensorDataset(data=torch.randn(100, 50))

# 2. Configure the data loader
loader = DiscreteTimeContrastiveDataLoader(batch_size=32, seed=42)
loader.lazy_init(dataset)
# 3. Set up the model components
latent_dim = 3
encoder = MLP(input_dim=dataset.observed_dim,
              output_dim=latent_dim,
              hidden_dim=180,
              num_layers=3)

# pick the dynamics model you want to use

# option 1: linear dynamics model
linear_dynamics = LinearDynamicsModel(dim=latent_dim)

# option 2: switchintg linear dynamics model
num_modes = 5
slds_dynamics = GumbelSLDS(
    linear_dynamics=LinearDynamicsModel(
        dim=latent_dim,
        num_systems=num_modes,
    ),
    switching_model=MSESwitchingModel(num_modes=num_modes,),
)

# option 3: define your own dynamics model
my_dynamics = MyDynamicsModel(...)

# 4. Create the solver
dynamics_model = linear_dynamics
solver = DynamicsContrastiveLearningSolver(
    model=encoder,
    dynamics_model=dynamics_model,
    optimizer=DCLAdamOptimizer(encoder_learning_rate=3e-4,
                               dynamics_learning_rate=3e-3),
    criterion=MseInfoNCE(
        temperature=1.0,
        infonce_type="infonce_full_denominator",
    ))

# 5. Train the model
solver.fit(loader)

# 6. Compute predictions
predictions = solver.predictions(loader)
embedding = predictions.embeddings
dynamics_predictions = predictions.dynamics
```

## 📚 Examples

The repository includes several example notebooks demonstrating DCL's capabilities:

1. [**Synthetic SLDS Data**](notebooks/demo_synthetic_slds.ipynb): Learn dynamics from synthetic Switching Linear Dynamical Systems (SLDS). This notebook demonstrates how DCL can identify and model multiple dynamical modes in simulated data.

2. **Data Analysis** (coming soon!): Application to real neural time-series data.

Check the `notebooks/` directory for these detailed examples.

## 🔄 Reproducing the results

1. Download the data from [here](https://nefeli.helmholtz-munich.de/records/tdfs2-kx054).
2. Follow the instructions in [`sweeps/README.md`](sweeps/README.md) to reproduce the results from Table 1.

## 📖 Reference

```bibtex
@inproceedings{
  laiz2025selfsupervised,
  title={Self-supervised contrastive learning performs non-linear system identification},
  author={Rodrigo Gonz{\'a}lez Laiz and Tobias Schmidt and Steffen Schneider},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=ONfWFluZBI}
}
```
