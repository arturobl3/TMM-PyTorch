from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Iterable
import torch
import torch.nn as nn

from .material import BaseMaterial                                             

LayerType = Literal["coh", "subs", "env"]     # “coherent”, “substrate”, “environment”

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
        """

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """ñ(λ) = √ε(λ)."""
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
    Standard thin-film layer: *material* + scalar *thickness*.

    Parameters
    ----------
    material : AbstractMaterial
        The optical material that furnishes ε(λ).
    thickness : float | torch.Tensor
        Physical thickness (meters or nm –  stick to *one* unit system).
    layer_type : {"coh", "subs", "env"}
        "coh"  – finite, coherently-interfering layer  
        "subs" – semi-infinite substrate  
        "env"  – ambient environment
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

        if layer_type not in ('coh', 'env', 'subs'):
            raise ValueError(
                f"layer_type must be 'coh', 'subs', or 'env', got {layer_type!r}"
            )
        self.layer_type: LayerType = layer_type

        # ------------------------ thickness handling ----------------------
        if layer_type == "coh":
            if thickness is None:
                raise ValueError("A coherent layer must have a thickness.")
            t = torch.as_tensor(thickness, dtype=material.dtype, device=material.device)
            if t.numel() != 1 or (t <= 0).any():
                raise ValueError("Thickness must be a positive scalar.")
            self.thickness = nn.Parameter(t, requires_grad=requires_grad)

        else:  # "subs" or "env"
            if thickness is not None:
                raise ValueError(f'"{layer_type}" layer must not specify thickness.')
            # store a *buffer* just so dtype/device are easy to query
            self.register_buffer(
                "thickness",
                torch.tensor(0.0, dtype=material.dtype, device=material.device),
            )

        

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:         
        return self.material.epsilon(wavelengths)

    def __repr__(self) -> str:
        base = super().__repr__()
        t_val = float(self.thickness.detach())
        return (f"{base[:-1]}, thickness={t_val:g}, type={self.layer_type})")