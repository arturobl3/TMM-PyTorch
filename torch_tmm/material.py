"""
Authors:
    Sergei Rodionov, Daniele Veraldi
Date:
    2025-06-09
License:
    MIT, Open Source

================================================================================
Module: material.py
================================================================================
Description:
    This module defines the base classes and implementations for optical materials
    used in multilayer simulations. Materials are constructed from one or more
    dispersion models that describe the wavelength-dependent complex permittivity ε(λ).

    Each material can be evaluated at arbitrary wavelengths to provide its dielectric
    response, either as permittivity or as a refractive index ñ(λ) = √ε(λ), and supports
    automatic differentiation for optimization workflows.

Key Components:
    - `BaseMaterial`: Abstract base class that defines the interface for material models.
    - `Material`: Concrete implementation that aggregates multiple dispersion models.

Conventions:
    - Dispersion models must implement an `epsilon()` method returning complex permittivity.
    - Units are assumed consistent across wavelength and dispersion models.
    - Materials automatically track device and dtype for compatibility with PyTorch operations.
    - Wavelengths are assumed to be in nanometers (nm)


Example:
    >>> import torch
    >>> from torch_tmm import Dispersion, Material
    >>> #Define materials
    >>> subs_mat = Material([Dispersion.Constant_epsilon(1)], name='Air')
    >>> env_mat = Material([Dispersion.Constant_epsilon(1.46**2)], name='Fused-silica')
    >>> metal_mat = Material([Dispersion.Lorentz(A=80, E0=0.845, Gamma=0.1),
                        Dispersion.Constant_epsilon(1)], name='Metal')
================================================================================
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .dispersion import BaseDispersion


class BaseMaterial(nn.Module, ABC):
    """
    Abstract base class to define material

    Attributes:
        name (str): the name of the implemented material
        dtype (torch.dtype): The data type for the PyTorch tensors.
        device (torch.device): The device (e.g., CPU or GPU) where the tensors are allocated.
    """

    def __init__(
        self,
        name: str | None = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize a BaseMaterial instance.

        Args:
            name (str): The name of the material implemented
            dtype (torch.dtype): The desired data type for tensor operations.
            device (torch.device): The device on which the tensors will be allocated.
        """
        super().__init__()

        # convenience mirrors (kept in sync by .to())
        self._dtype = dtype
        self._device = device
        self.name = name

        # move all sub-modules to the requested dtype/device
        self.to(dtype=dtype, device=device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        """
        Mirror nn.Module.to and keep internal mirrors in sync.
        """
        ret = super().to(*args, **kwargs)

        # adopt dtype/device of the first parameter/buffer
        try:
            p = next(self.parameters())
            self._dtype, self._device = p.dtype, p.device
        except StopIteration:  # parameter-less material
            self._dtype = kwargs.get("dtype", self._dtype)
            self._device = kwargs.get("device", self._device)

        return ret

    def _sync_dtype_device(self) -> None:
        """Update private mirrors from the first parameter or buffer."""
        try:
            p = next(self.parameters())
        except StopIteration:
            try:
                p = next(self.buffers())
            except StopIteration:
                return
        self._dtype, self._device = p.dtype, p.device

    def _apply(self, fn):
        # let nn.Module move / cast all tensors
        out = super()._apply(fn)
        # then refresh mirrors so .dtype / .device stay correct
        self._sync_dtype_device()
        return out

    # -------------------------- public API -------------------------------
    @abstractmethod
    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """Return ε(λ) as a complex tensor."""

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """ñ(λ) = √ε(λ).  Provided once here for every subclass."""
        return torch.sqrt(self.epsilon(wavelengths))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, dtype={self.dtype}, device={self.device})"
        )


class Material(BaseMaterial):
    """
    Material aggregates multiple dispersion models to represent an optical material.

    This class is designed to combine the contributions of several dispersion models
    to compute the overall refractive index of a material. It is particularly useful
    in optical simulations and thin film optimization where the optical response
    may result from several dispersion effects.

    Attributes:
        dispersion (List[Dispersion]): A list of dispersion model instances. Each instance
            must implement a `getRefractiveIndex()` method.
        name (str): the name of the implemented material
        dtype (torch.dtype): The data type for the PyTorch tensors.
        device (torch.device): The device (e.g., CPU or GPU) where the tensors are allocated.
    """

    def __init__(
        self,
        dispersion: list[BaseDispersion],
        name: str | None = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
        *,
        requires_grad: bool = False,
    ) -> None:
        """
        Initialize a BaseMaterial instance.

        Args:
            dispersion (List[Dispersion]): A list of dispersion model instances.
            name (str): The name of the material implemented
            dtype (torch.dtype): The desired data type for tensor operations.
            device (torch.device): The device on which the tensors will be allocated.
            requires_grad : bool, default ``False``
        If ``True`` the material parameters become optimizible.
        """
        super().__init__()

        # register children so that .parameters(), .to(), etc. work
        self.dispersion = nn.ModuleList(dispersion)

        # convenience mirrors (kept in sync by .to())
        self._dtype = dtype
        self._device = device
        self.name = name

        # move all sub-modules to the requested dtype/device
        self._smart_to(dtype=dtype, device=device)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def _smart_to(self, dtype=None, device=None):
        """
        Intelligent dtype/device conversion that preserves complex buffers.

        This method handles the dtype conversion more carefully than the default
        nn.Module.to() method. It:
        1. Converts all tensors to the target device
        2. Converts parameters to the target dtype
        3. Preserves complex dtypes for buffers (to avoid data loss)

        Args:
            dtype: Target dtype for parameters (buffers with complex data are preserved)
            device: Target device for all tensors
        """
        # Handle device movement first (this is safe for all tensors)
        if device is not None:
            super().to(device=device)

        # Handle dtype conversion with complex preservation
        if dtype is not None:
            # Convert parameters to target dtype
            for param in self.parameters():
                if param.dtype != dtype:
                    param.data = param.data.to(dtype=dtype)

            # For buffers, only convert if not complex or if target is also complex
            for name, buffer in self.named_buffers():
                if buffer.dtype.is_complex:
                    # Preserve complex buffers - don't convert to real dtypes
                    if dtype.is_complex:
                        buffer.data = buffer.data.to(dtype=dtype)
                    # If target is real, keep complex buffer as-is (preserve data)
                    else:
                        continue
                else:
                    # Non-complex buffers can be safely converted
                    buffer.data = buffer.data.to(dtype=dtype)

        # Update internal mirrors
        self._dtype = dtype if dtype is not None else self._dtype
        self._device = device if device is not None else self._device

    def to(self, *args, **kwargs):
        """
        Override to() method to use smart dtype handling.

        This preserves complex buffers while allowing dtype conversion of parameters.
        """
        # Parse arguments like nn.Module.to() does
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        # Handle positional arguments
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            elif isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], torch.Tensor):
                dtype = args[0].dtype
                device = args[0].device
        elif len(args) == 2:
            device, dtype = args

        # Use our smart conversion
        self._smart_to(dtype=dtype, device=device)

        return self

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the overall epsilon of the material.

        This method calculates the material's epsilon by summing the dielectric permittivity
        contributions from each dispersion model in the `dispersion` list. The summation
        is performed element-wise over a tensor that spans the specified number of wavelengths.

        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns:
            torch.tensor: A 1D tensor of shape (num_wavelength,) representing the computed
                          refractive index at each wavelength.
        """
        if not self.dispersion:
            raise RuntimeError("Material has no dispersion models.")

        # input validation via the first dispersion helper
        wavelengths = self.dispersion[0]._prepare_wavelengths(wavelengths)

        eps_list = [d.epsilon(wavelengths) for d in self.dispersion]
        return torch.stack(eps_list, dim=0).sum(dim=0)

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the overall refractive index of the material.

        This method calculates the material's refractive index.

        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns:
            torch.tensor: A 1D tensor of shape (num_wavelength,) representing the computed
                          refractive index at each wavelength.
        """
        return torch.sqrt(self.epsilon(wavelengths))

    def __repr__(self) -> str:
        """
        Return a string representation of the Material instance.

        Returns:
            str: A string summarizing the Layer.
        """

        dispersion_repr = (
            f"[{', '.join(repr(dispersion) for dispersion in self.dispersion)}]"
        )
        return (
            f"Material(\n"
            f"  Name: {self.name},\n"
            f"  Dispersions: {dispersion_repr},\n (n={len(self.dispersion)} dispersions)"
        )
