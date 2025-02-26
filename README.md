# T_matrix: A PyTorch-based Transfer Matrix Method Library

This repository implements the `T_matrix` class for computing transfer matrices in thin film optics using the Transfer Matrix Method (TMM). The module leverages PyTorch tensors to perform vectorized, high-performance computations on both CPU and GPU, and it is fully compatible with PyTorch's automatic differentiation (autograd).

## Key Features

- **Vectorized Computation:** All methods operate on PyTorch tensors for efficient, batch-style calculations.
- **GPU Support:** Run computations on GPU for accelerated performance.
- **Autograd Compatibility:** Seamlessly integrate with PyTorch's autograd for gradient-based optimization.
- **Modular Design:** 
  - **coherent_layer:** Computes the overall transfer matrix for a single coherent layer (surrounded by air) over a range of wavelengths and angles.
  - **interface_s:** Computes the interface matrix between two media for s-polarization.
  - **interface_p:** Computes the interface matrix between two media for p-polarization.
  - **propagation_coherent:** Computes the propagation transfer matrix through a layer.

## Conventions

- **Propagation Direction:** Calculations assume propagation from left to right.
- **Refractive Index:** Defined as `n_real + 1j * n_imm`.
- **Units:** Wavelengths and thicknesses must be defined in the same units (e.g., m, nm, or µm).
- **Angles:** Specified in degrees, in the range [0, 90).

## Usage Guidelines

For cases involving high complex refractive indices or very thick layers, computational errors may occur when using `dtype=torch.complex64` or `dtype=torch.complex32`. In such cases, it is recommended to use `dtype=torch.complex128` for improved numerical stability.

## Example

Below is a sample script demonstrating how to use the `T_matrix` class:

```python
from torch_tmm import Model, BaseLayer, BaseMaterial
from torch_tmm.dispersion import Constant_epsilon
import torch

# define the type and device used during the calculation
dtype=torch.complex64
device=torch.device('cpu')

# define wavelenghts and incidence angle
wavelengths = torch.linspace(400, 800, 401)
angles = torch.linspace(0, 89, 90)

# define dispersions describing the materials used
env_disp = [Constant_epsilon(1+0j, dtype, device)]
layer_disp = [Constant_epsilon(3+0.2j, dtype=dtype, device=device)]
subs_disp = [Constant_epsilon(5+0j, dtype, device)]

# define materials to be simulated
env_mat = BaseMaterial(env_disp,name = 'air',dtype= dtype,device= device)
layer_mat = BaseMaterial(layer_disp,dtype= dtype,device= device)
subs_mat = BaseMaterial(subs_disp,name = 'glass',dtype= dtype, device=device)

# define layers
env = BaseLayer(env_mat, thickness=0, LayerType='env')
layer = BaseLayer(layer_mat, thickness=torch.tensor(10,dtype=dtype, device=device), LayerType='coh')
subs = BaseLayer(subs_mat, thickness=0, LayerType='subs')

#define model and perform calculations
model = Model(env, [layer], subs, dtype, device)
results= model.evaluate(wavelengths, angles)
