## TMM-PyTorch

A PyTorch-based library for optical multilayer stacks simulation using the Transfer-Matrix Method (TMM). TMM-PyTorch combines complex-valued dispersion models, GPU acceleration, and PyTorch’s autograd to let you both **compute** reflectance/transmission spectra and **fit** layer or material parameters end-to-end.

**The framework is still under development and some functionality can be abscent** 

---

## Key Features

* **Modular Dispersion Models**

  * Abstract base class `BaseDispersion` plus built-in models: `ConstantEpsilon`, `Lorentz`, `Cauchy`, `Tauc–Lorentz`, etc.
  * Easily extend by subclassing `BaseDispersion`.

* **Materials & Layers**

  * `BaseMaterial` / `Material` aggregate dispersions.
  * `BaseLayer` / `Layer` wrap materials + thickness + layer type (`coh`/`env`/`subs`).
  * Public API:

    ```python
    eps = material.epsilon(wavelengths)      # complex ε(λ)
    n   = layer.refractive_index(wavelengths)
    ```

* **Full-stack Model**

  * `Model(env, structure, subs)` orchestrates:

    1. Environment layer (incident medium, `env`).
    2. Intermediate coherent layers (`structure`).
    3. Substrate layer (transmission medium, `subs`).
  * Computes transfer matrices for s/p polarizations over any wavelength × angle grid.

* **GPU & Mixed Precision**

  * Call `.to(device, dtype)` on any dispersion/material/layer/model:

    ```python
    model = model.to("cuda", dtype=torch.float64)
    ```
  * Real parameters stay `float32⇄float64`; outputs auto-promote to `complex64⇄complex128`.
  * Works with PyTorch AMP and `torch.compile(model)` for additional speedups.

* **End-to-End Differentiable**

  * All thicknesses and dispersion parameters are `nn.Parameter`.
  * Use standard PyTorch optimizers (Adam, SGD, etc.) to fit to measured spectra.

---

## Installation

```bash
pip install git+https://github.com/RodionovSA/torch_tmm.git
```

Requires **Python 3.10+**, **Numpy 1.21+**, **PyTorch 2.0+** (CPU or CUDA).

---

## Project Structure

```
torch_tmm/                  # core package
├── t_matrix.py             # T_matrix: interface & propagation kernels
├── dispersion.py           # BaseDispersion + models (Constant, Lorentz, etc.)
├── material.py             # BaseMaterial / Material classes
├── layer.py                # BaseLayer / Layer classes
└── model.py                # high-level Model (env+structure+subs)
└── optical_calculator.py   # OpticalCalculator
tmm_tests/                  # unit & sanity tests
README.md                   # this file
setup.py
LICENSE
```

Exports:

```python
from torch_tmm import (
    T_matrix,
    BaseDispersion, ConstantEpsilon, Lorentz, …,
    BaseMaterial, Material,
    BaseLayer, Layer,
    Model,
)
```

---

## Quick Start

```python
import torch
from torch_tmm import ConstantEpsilon, Material, Layer, Model

# 1D wavelength & angle arrays
wls = torch.linspace(400, 800, 401)     # nm
ths = torch.linspace(0, 80, 81)         # degrees

# Dispersion models
disp_env   = ConstantEpsilon(1.0)       # air
disp_film  = Lorentz(A=1.9, E0=3.2, Gamma=0.15)
disp_subs  = ConstantEpsilon(2.5)       # substrate

# Materials
env_mat   = Material([disp_env],   name="Air")
film_mat  = Material([disp_film],  name="Film")
subs_mat  = Material([disp_subs],  name="Substrate")

# Layers
env_layer  = Layer(env_mat,  layer_type="env")
film_layer = Layer(film_mat, layer_type="coh",  thickness=500e-9)
subs_layer = Layer(subs_mat, layer_type="subs")

# Build model and move to GPU
model = Model(env_layer, [film_layer], subs_layer)
model = model.to("cuda", dtype=torch.float32)

# Compute s-polarization transmission
results = model(wls.to(model.device), ths.to(model.device))
```

---

## Optimization Example

```python
import torch.nn as nn

# Suppose T_target is your measured spectrum
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    out = model(wls, ths).transmission("s")
    loss = criterion(out, T_target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4e}, film thickness={film_layer.thickness.item():.4e}")
```

---

## GPU & Performance Tips

* Use **`torch.compile(model)`** for JIT optimizations.
* Employ **automatic mixed precision** (AMP) for faster kernels.
* Ensure batched matmuls (`@`) drive throughput; avoid Python loops in hot paths.

---

## Contributing

1. Fork & clone
2. Create a feature branch
3. Add tests under `tests/`
4. Submit a pull request

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
