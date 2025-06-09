"""
Authors:
    Sergei Rodionov, Daniele Veraldi
Date:
    2025-06-09
License:
    MIT, Open Source

================================================================================
Module: layer.py
================================================================================
Description:
    This module defines the data structures used to describe individual optical layers 
    in a multilayer stack. Each layer wraps an optical material and defines how the 
    material is spatially arranged—either as a finite-thickness film or a 
    semi-infinite medium (environment or substrate).

    Layers expose a unified interface to compute complex permittivity ε(λ) and 
    refractive index ñ(λ) for use in transfer matrix calculations.

Key Components:
    - `BaseLayer`: Abstract base class for all layers. Defines the optical interface.
    - `Layer`: Concrete implementation for coherent and semi-infinite layers.
      Associates a material model with a thickness and layer type.

Conventions:
    - Coherent layers (`'coh'`) must define a non-negative thickness.
    - Semi-infinite layers (`'semi-inf'`) must omit the thickness.
    - Thickness units must be consistent across layers (nm or meters).
    - All tensors will inherit `dtype` and `device` from the associated material.
    
Example:
    >>> import torch
    >>> from torch_tmm import Dispersion, Material, Layer
    >>> #Define materials
    >>> subs_mat = Material([Dispersion.Constant_epsilon(1)], name='Air')
    >>> env_mat = Material([Dispersion.Constant_epsilon(1.46**2)], name='Fused-silica')
    >>> metal_mat = Material([Dispersion.Lorentz(A=80, E0=0.845, Gamma=0.1),
                        Dispersion.Constant_epsilon(1)], name='Metal')
    >>> #Define layers
    >>> env = Layer(env_mat, layer_type='semi-inf')
    >>> subs = Layer(subs_mat, layer_type='semi-inf')
    >>> metal = Layer(metal_mat, layer_type='coh', thickness=25) #nm
================================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, Iterable
import torch
import torch.nn as nn
from .material import BaseMaterial                                             

LayerType = Literal["coh", "semi-inf"]     # “coherent”, “semi-infinite”

class BaseLayer(nn.Module, ABC):
    """
    Common API for all layer types.

    Every concrete subclass must implement :meth:`epsilon`.  A default
    implementation of :meth:`refractive_index` is provided here to avoid
    duplication.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name

    # ---------- optical response -----------------------------------------
    @abstractmethod
    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Complex permittivity ε(λ).

        *Shape*: ``(..., len(wavelengths))``.  The method **must** accept
        inputs on any device / dtype that the layer (and its sub-modules)
        currently live on.

        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor). 
        """

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        ñ(λ) = √ε(λ)

        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor). 
        """
        return torch.sqrt(self.epsilon(wavelengths))

    # ---------- convenience mirrors --------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        # use the dtype of the first parameter/buffer if present
        for p in self.parameters():
            return p.dtype
        for b in self.buffers():
            return b.dtype
        return torch.get_default_dtype()

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    # ---------- representation -------------------------------------------
    def extra_repr(self) -> str:
        return f"name={self.name!r}, dtype={self.dtype}, device={self.device}"


class Layer(BaseLayer):
    """
    Standard thin-film or semi-infinite layer: *material* + scalar *thickness*.

    Parameters
    ----------
    material : AbstractMaterial
        The optical material that furnishes ε(λ).
    thickness : float | torch.Tensor
        Physical thickness (meters or nm –  stick to *one* unit system).
    layer_type : {"coh", "semi-inf"}
        "coh"  – finite, coherently-interfering layer  
        "semi-inf" – semi-infinite substrate  
    requires_grad : bool, default ``False``
        If ``True`` the thickness becomes an optimisable parameter.
    """

    def __init__(
        self,
        material: BaseMaterial,
        *,
        layer_type: LayerType = "coh",
        thickness: float | torch.Tensor | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.material = material      

        if layer_type not in ('coh', 'semi-inf'):
            raise ValueError(
                f"layer_type must be 'coh', or 'semi-inf', got {layer_type!r}"
            )
        self.layer_type: LayerType = layer_type

        # ------------------------ thickness handling ----------------------
        if layer_type == "coh":
            if thickness is None:
                raise ValueError("A coherent layer must have a thickness.")
            t = torch.as_tensor(thickness, dtype=material.dtype, device=material.device)
            if t.numel() != 1 or (t < 0).any():
                raise ValueError("Thickness must be a non-negative scalar.")
            self.thickness = nn.Parameter(t, requires_grad=requires_grad)

        else:  # "semi-inf"
            if thickness is not None:
                raise ValueError(f'"{layer_type}" layer must not specify thickness.')
            # store a *buffer* just so dtype/device are easy to query
            self.register_buffer(
                "thickness",
                torch.tensor(0.0, dtype=material.dtype, device=material.device),
            )

        

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor: 
        """
        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor). 
        """        
        return self.material.epsilon(wavelengths)

    def __repr__(self) -> str:
        base = super().__repr__()
        t_val = float(self.thickness.detach())
        return (f"{base[:-1]}, thickness={t_val:g}, type={self.layer_type})")