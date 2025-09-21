"""
Authors:
    Sergei Rodionov, Daniele Veraldi
Date:
    2025-06-09
License:
    MIT, Open Source

================================================================================
Module: dispersion.py
================================================================================
Description:
    This module defines dispersion models for simulating the optical
    response of materials. Each dispersion model implements wavelength-dependent
    complex permittivity ε(λ) and refractive index ñ(λ).

    The dispersion models support automatic differentiation, device/dtype management,
    and composability within broader optical simulation frameworks.

Key Components:
    - `BaseDispersion`       : Abstract base class for dispersion models
    - `Constant_epsilon`     : Constant (wavelength-independent) dielectric permittivity
    - `Lorentz`              : Classical Lorentz oscillator model
    - `Cauchy`               : Polynomial model for transparent materials (real + imaginary parts)
    - `TaucLorentz`          : Amorphous semiconductor model combining Lorentz and bandgap behavior

Conventions:
    - Wavelengths are assumed to be in nanometers (nm)
    - All models operate on `torch.Tensor` inputs with support for batched evaluation
    - All parameters are `torch.nn.Parameter`, enabling learnable optical models
    - Gradient support is maintained for inverse design and optimization

Example:
    >>> import torch
    >>> from torch_tmm import Dispersion
    >>> wvl = torch.linspace(400, 800, 801)
    >>> #Define dispersions
    >>> const_eps = Dispersion.Constant_epsilon(1)
    >>> lorentz = Dispersion.Lorentz(A=80, E0=0.845, Gamma=0.1)
    >>> epsilon = const_eps.epsilon(wvl) + lorentz.epsilon(wvl)
================================================================================
"""

from typing import Union
from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn


class BaseDispersion(nn.Module, ABC):
    """
    Abstract base class to define dispersion models for materials

    Attributes:
        dtype (torch.dtype): The data type for the torch tensor (e.g., torch.float32).
        device (torch.device): The device on which to allocate tensors (e.g., CPU or GPU).
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initializes the base dispersion model with the specified data type and device.

        Args:
            dtype (torch.dtype, optional): The data type for model parameters (default: torch.float32).
            device (torch.device, optional): The computation device (default: CPU).

        This constructor ensures that all dispersion models derived from this class will operate with
        a consistent data type and device.
        """
        super().__init__()
        super().__setattr__("_dtype", dtype)
        super().__setattr__("_device", device)

    @abstractmethod
    def epsilon(self, wavelengths: torch.Tensor, *args, **kwargs) -> torch.tensor:
        """method to calculate the dielectric constant of a material"""

    @abstractmethod
    def refractive_index(
        self, wavelengths: torch.Tensor, *args, **kwargs
    ) -> torch.tensor:
        """method to calculate the refractive index of a material"""

    @abstractmethod
    def __repr__(self) -> str:
        """method to return the dispersion parameters"""

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def _validate_value(
        self,
        name: str,
        value: Union[float, int, np.ndarray, torch.Tensor, torch.nn.Parameter],
    ) -> None:
        """
        Validates the provided value.

        Raises:
            ValueError: If the value is of a disallowed type (e.g., bool, str, list)
                        or if it's a tensor/parameter that doesn't contain exactly one element.
        """
        if isinstance(value, bool):
            raise ValueError(f"Parameter '{name}' cannot be of type bool.")

        if isinstance(value, (str, list, tuple, dict)):
            raise ValueError(
                f"Parameter '{name}' must be a single numeric value, not {type(value).__name__}."
            )

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    f"Parameter '{name}' must be a single numeric value; got numpy array with {value.size} elements."
                )

        if isinstance(value, (torch.Tensor, nn.Parameter)):
            # If it's a tensor or parameter, ensure it has exactly one element.
            tensor_val = value.data if isinstance(value, nn.Parameter) else value
            if tensor_val.numel() != 1:
                raise ValueError(
                    f"Parameter '{name}' must be a single numeric value; got tensor with {tensor_val.numel()} elements."
                )

    def _convert_value(
        self, value: Union[float, int, np.ndarray, torch.Tensor, torch.nn.Parameter]
    ) -> nn.Parameter:
        """
        Converts a value (float, int, torch.Tensor, or nn.Parameter) into a torch.nn.Parameter
        with the current dtype and device.
        """
        if not isinstance(value, torch.Tensor):  # If not torch.Tensor convert to it
            value = torch.tensor(value, dtype=self._dtype, device=self._device)
        else:
            value = value.to(dtype=self._dtype, device=self._device)
        return nn.Parameter(value, requires_grad=False)

    def __setattr__(
        self,
        name: str,
        value: Union[float, int, np.ndarray, torch.Tensor, torch.nn.Parameter],
    ) -> None:
        """
        Overrides attribute assignment so that any numerical parameter provided (float, int, Tensor)
        is automatically converted to a torch.nn.Parameter registered in the module.

        Note: The conversion is done only once during assignment. Later updates of device/dtype are handled via `.to()`.
        """
        # Skip PyTorch's built-in attributes AND TMM's internal attributes
        pytorch_builtin_attrs = {
            "_dtype",
            "_device",  # TMM internal attributes
            "training",  # PyTorch training mode
            "_parameters",
            "_buffers",  # PyTorch module internals
            "_non_persistent_buffers_set",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_modules",  # PyTorch submodules
            "_hc_over_e",  # TMM buffer for physical constants
        }

        if name in pytorch_builtin_attrs:
            super().__setattr__(name, value)
            return

        # user-facing attributes go through validation/conversion
        self._validate_value(name, value)
        super().__setattr__(name, self._convert_value(value))

    def to(self, *args, **kwargs):
        """
        Mirrors nn.Module.to **and** synchronises self._dtype/_device
        with wherever the parameters actually ended up.
        """
        # Let nn.Module do the heavy lifting first (handles every call style)
        ret = super().to(*args, **kwargs)

        # Pick the dtype/device of the first parameter or buffer
        try:
            p = next(self.parameters())
        except StopIteration:
            p = None
        if p is None:  # parameter-less module
            self._dtype = kwargs.get("dtype", self._dtype)
            self._device = kwargs.get("device", self._device)
        else:
            self._dtype = p.dtype
            self._device = p.device
        return ret  # keep nn.Module semantics (returns self)

    @staticmethod
    def _as_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
        """
        Promote a real floating dtype to the minimal matching complex dtype.
        float16/32/bfloat16 ➜ complex64, float64 ➜ complex128.
        (If `real_dtype` is already complex, it is returned unchanged.)
        """
        if real_dtype.is_complex:
            return real_dtype
        if real_dtype == torch.float64:
            return torch.complex128
        if real_dtype.is_floating_point:
            return torch.complex64
        raise TypeError(f"{real_dtype} is not a floating dtype.")

    def _prepare_wavelengths(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        • Ensures wavelengths > 0 (no zero / negative wavelengths).
        • Casts to this module’s real dtype & device.
        • Keeps gradient flow intact (no `detach` / `clone`).
        """
        if not torch.is_floating_point(wavelengths):
            raise TypeError("`wavelength` must be a floating tensor.")

        if (wavelengths <= 0).any():
            bad = wavelengths[wavelengths <= 0]
            raise ValueError(f"Wavelengths must be positive; got {bad.cpu().tolist()}")

        return wavelengths.to(dtype=self.dtype, device=self.device)

    def _sync_dtype_device(self):
        try:
            p = next(self.parameters())
            self._dtype, self._device = p.dtype, p.device
        except StopIteration:
            pass

    def _apply(self, fn):
        out = super()._apply(fn)  # let Module move the tensors
        self._sync_dtype_device()  # then sync our mirrors
        return out


class Constant_epsilon(BaseDispersion):
    """
    A dispersion model with a constant (flat) dielectric permittivity.
    This class implements a dispersion model in which the dielectric permittivity remains
    constant across all wavelengths. It inherits from the Dispersion base class.

    Attributes:
        epsilon_const (torch.nn.Parameter): The constant dielectric permittivity value.
    """

    def __init__(
        self,
        epsilon_const: torch.nn.Parameter,
    ) -> None:
        """
        Initialize the Constant_epsilon instance.

        Args:
            epsilon_const (torch.nn.Parameter): The constant dielectric permittivity value.
        """
        super().__init__()
        self.epsilon_const = epsilon_const

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the (generally complex) refractive index **n(λ)** for each input wavelength.

        The constant-ε model yields
            n(λ) = √ε ,
        so the value is the same for every λ, but the tensor you receive has
        the same shape and device as `wavelengths`, allowing easy broadcasting
        or later concatenation with wavelength-dependent models.

        Parameters
        ----------
        wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
            A tensor of identical shape to `wavelengths`, containing the
            refractive index.
        """
        return torch.sqrt(self.epsilon(wavelengths))

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Dielectric permittivity **ε(λ)** for the constant-ε model.

        Parameters
        ----------
        wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
            Complex tensor filled with the constant permittivity value,
            matching `wavelengths` in shape, device, and gradient behaviour.
        """
        wavelengths = self._prepare_wavelengths(wavelengths)
        c_dtype = self._as_complex_dtype(self.dtype)

        eps_const = self.epsilon_const.to(dtype=c_dtype, device=self.device)
        epsilon = eps_const * torch.ones_like(
            wavelengths, dtype=c_dtype, device=self.device
        )

        return epsilon

    def __repr__(self) -> str:
        """
        Return a string representation of the dispersion instance.

        Returns:
            str: A string summarizing the dispersion.
        """

        return f"Constant Dispersion: {self.epsilon_const}"


class Lorentz(BaseDispersion):
    """
    Implements the Lorentz oscillator model for optical dispersion.
    This class computes the electric permittivity and refractive index based on the Lorentz oscillator model.

    Attributes:
        A (torch.nn.Parameter): Oscillator amplitude, eV**2.
        E0 (torch.nn.Parameter): Resonance energy, eV.
        C (torch.nn.Parameter): Damping coefficient, eV.
    """

    _hc_over_e: torch.Tensor  # pre-computed in __init__ for speed

    def __init__(
        self,
        A: torch.nn.Parameter,
        E0: torch.nn.Parameter,
        Gamma: torch.nn.Parameter,
    ) -> None:
        """
        Initialize the Lorentz dispersion model with given parameters.
        Args:
            A (torch.nn.Parameter): Oscillator amplitude, eV**2.
            E0 (torch.nn.Parameter): Resonance energy, eV.
            Gamma (torch.nn.Parameter): Damping coefficient, eV.
        """
        super().__init__()
        self.A = A
        self.E0 = E0
        self.Gamma = Gamma

        hc_over_e = 1.2398419843320026e3  # (h·c/e) in  eV·nm
        self.register_buffer(
            "_hc_over_e", torch.tensor(hc_over_e, dtype=self.dtype, device=self.device)
        )

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Complex refractive index **n(λ)** derived from the Lorentz-oscillator
        permittivity.

        Given the electric permittivity ε(λ) produced by
        :meth:`~Lorentz.epsilon`, the refractive index is

            n(λ) = √ε(λ)

        with the principal square-root branch.

        Parameters
        ----------
        wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
            Complex tensor containing the refractive index at each provided
            wavelength.
        """
        return torch.sqrt(self.epsilon(wavelengths))

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex dielectric permittivity **ε(λ)** using the Lorentz oscillator model.
        The electric permittivity **ε(λ)** is computed using the formula:
            ε = A / (E0^2 - E^2 - i * Gamma * E)
        where E is the photon energy calculated as:
            E = (h * c / e) / (wavelengths)
        Constants:
            - h (Planck constant): 6.62607015e-34 J·s
            - c (Speed of light): 299792458 m/s
            - e (Elementary charge): 1.60217663e-19 C

        Parameters
        ----------
        wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
           The computed complex electric permittivity.

        """
        wavelengths = self._prepare_wavelengths(wavelengths)
        E = self._hc_over_e.to(self.dtype) / wavelengths

        c_dtype = self._as_complex_dtype(self.dtype)
        E = E.to(dtype=c_dtype)
        A = self.A.to(dtype=c_dtype)
        E0 = self.E0.to(dtype=c_dtype)
        Gamma = self.Gamma.to(dtype=c_dtype)

        # Lorentz electric permittivity calculation
        epsilon = A / (E0**2 - E**2 - 1j * Gamma * E)

        return epsilon

    def __repr__(self) -> str:
        """
        Return a string representation of the dispersion instance.

        Returns:
            str: A string summarizing the dispersion.
        """

        return f"Lorentz Dispersion(Coefficients (A,E0,Gamma):{self.A, self.E0, self.Gamma})"


class Cauchy(BaseDispersion):
    """
    Implements the Cauchy dispersion model for optical materials.

    This model expresses the complex refractive index as a function of wavelength using
    the Cauchy equations for both the real and imaginary parts. It employs six coefficients,
    provided as torch.nn.Parameter objects, which are scaled appropriately in the formulas.

    The real part (n) and the imaginary part (k) of the refractive index are computed as:
        n = A + (1e4 * B) / wavelength² + (1e9 * C) / wavelength⁴
        k = D + (1e4 * E) / wavelength² + (1e9 * F) / wavelength⁴
    so that the complex refractive index is:
        ñ = n + i * k

    Attributes:
        A, B, C (torch.nn.Parameter): Coefficients for the real part of the refractive index.
        D, E, F (torch.nn.Parameter): Coefficients for the imaginary part (extinction) of the refractive index.
    """

    def __init__(
        self,
        A: torch.nn.Parameter,
        B: torch.nn.Parameter = 0,
        C: torch.nn.Parameter = 0,
        D: torch.nn.Parameter = 0,
        E: torch.nn.Parameter = 0,
        F: torch.nn.Parameter = 0,
    ) -> None:
        """
        Initialize the Cauchy dispersion model with specified coefficients.

        Args:
            A (torch.nn.Parameter): Coefficient for the constant term in the real part.
            B (torch.nn.Parameter): Coefficient for the 1/wavelength² term in the real part.
            C (torch.nn.Parameter): Coefficient for the 1/wavelength⁴ term in the real part.
            D (torch.nn.Parameter): Coefficient for the constant term in the imaginary part.
            E (torch.nn.Parameter): Coefficient for the 1/wavelength² term in the imaginary part.
            F (torch.nn.Parameter): Coefficient for the 1/wavelength⁴ term in the imaginary part.
        """
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Cauchy-type complex refractive index

            n(λ) = A + 1e4·B / λ² + 1e9·C / λ⁴
            k(λ) = D + 1e4·E / λ² + 1e9·F / λ⁴
            ñ(λ) = n(λ) + i·k(λ)

        Parameters
        ----------
        wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
            Complex refractive index at each λ;
        """
        wavelengths = self._prepare_wavelengths(wavelengths)
        c_dtype = self._as_complex_dtype(self.dtype)
        wavelengths = wavelengths.to(c_dtype)

        A = self.A.to(dtype=c_dtype)
        B = self.B.to(dtype=c_dtype)
        C = self.C.to(dtype=c_dtype)
        D = self.D.to(dtype=c_dtype)
        E = self.E.to(dtype=c_dtype)
        F = self.F.to(dtype=c_dtype)

        n = A + 1e4 * B / wavelengths**2 + 1e9 * C / wavelengths**4
        k = D + 1e4 * E / wavelengths**2 + 1e9 * F / wavelengths**4
        return n + 1j * k

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Complex electric permittivity **ε(λ)** for this material model.

        The permittivity is obtained directly from the complex refractive
        index **ñ(λ)** returned by :meth:`refractive_index` via

            ε(λ) = ñ(λ)² .

        Parameters
        ----------
        wavelengths : torch.Tensor
           Wavelengths in **nanometres** (positive, floating tensor).

        Returns
        -------
        torch.Tensor
            Complex tensor containing ε(λ) for each supplied wavelength.
        """

        return (self.refractive_index(wavelengths)) ** 2

    def __repr__(self) -> str:
        """
        Return a string representation of the Cauchy dispersion instance.

        Returns:
            str: A string summarizing the Cauchy dispersion model with its coefficients,
                 data type, and device.
        """

        return f"Cauchy Dispersion(Coefficients(A,B,C,D,E,F):{[self.A, self.B, self.C, self.D, self.E, self.F]})"


class TaucLorentz(BaseDispersion):
    """
    TaucLorentz dispersion model for optical materials.

    This class implements the Tauc-Lorentz model to describe the complex dielectric function
    (electric permittivity) of amorphous semiconductors. The model is characterized by a set of
    coefficients that define the optical response, including the optical band gap (Eg), amplitude (A),
    resonance energy (E0), and broadening parameter (C).

    The complex dielectric function is given by:
        ε(E) = ε_r(E) + i·ε_i(E)
    where the imaginary part ε_i(E) is nonzero only for photon energies E greater than the band gap Eg,
    and the real part ε_r(E) is computed via a Kramers-Kronig transformation involving logarithmic and
    arctan terms.

    Attributes:
        Eg (torch.nn.Parameter): Optical band gap energy.
        A (torch.nn.Parameter): Amplitude of the transition.
        E0 (torch.nn.Parameter): Resonance energy.
        C (torch.nn.Parameter): Broadening (damping) parameter.
    """

    _hc_over_e: torch.Tensor  # pre-computed in __init__ for speed

    def __init__(
        self,
        Eg: torch.nn.Parameter,
        A: torch.nn.Parameter,
        E0: torch.nn.Parameter,
        Gamma: torch.nn.Parameter,
    ) -> None:
        """
        Initialize the TaucLorentz model with the specified parameters.

        Args:
            Eg (torch.nn.Parameter): Optical band gap energy.
            A (torch.nn.Parameter): Amplitude of the transition.
            E0 (torch.nn.Parameter): Resonance energy.
            Gamma (torch.nn.Parameter): Broadening (damping) parameter.
        """
        super().__init__()
        self.Eg = Eg
        self.A = A
        self.E0 = E0
        self.Gamma = Gamma

        hc_over_e = 1.2398419843320026e3  # (h·c/e) in  eV·nm
        self.register_buffer(
            "_hc_over_e", torch.tensor(hc_over_e, dtype=self.dtype, device=self.device)
        )

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex refractive index at the given wavelengths.

        Args:
            wavelengths : torch.Tensor
           Wavelengths in **nanometres** (positive, floating tensor).
        Returns:
            torch.Tensor: Complex refractive index evaluated at the specified wavelengths.
        """
        return torch.sqrt(self.epsilon(wavelengths))

    def epsilon(self, wavelengths: torch.Tensor):
        """
        Compute the complex dielectric function using the Tauc-Lorentz model.

        The photon energy E is calculated from the wavelengths (meters) using:
            E = (h * c / e) / (wavelengths)
        where:
            - h (Planck constant) = 6.62607015e-34 J·s,
            - c (speed of light) = 299792458 m/s,
            - e (elementary charge) = 1.60217663e-19 C.

        The model parameters are unpacked as:
            Eg: Optical band gap energy.
            A: Amplitude of the transition.
            E0: Resonance energy.
            Gamma: Broadening (damping) parameter.

        For photon energies E greater than Eg, the imaginary part ε_i is computed by:
            ε_i = (1/E) * (A * E0 * Gamma * (E - Eg)^2) / ((E^2 - E0^2)^2 + Gamma^2 * E^2)
        For E ≤ Eg, ε_i is set to 0.

        The real part ε_r is obtained from several contributions (epsilon_r1 through epsilon_r5)
        that involve logarithmic and arctan functions to satisfy the Kramers-Kronig relations.

        Finally, any NaN values in ε_r and ε_i are replaced by zero, and the complex dielectric function is:
            ε = ε_r + i * ε_i

        Args:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).
        Returns:
            torch.Tensor: Complex dielectric function evaluated at the specified wavelengths.
        """

        wavelengths = self._prepare_wavelengths(wavelengths)  # nm → validated
        E = self._hc_over_e.to(self.dtype) / wavelengths  # eV   (real)
        mask = E > self.Eg

        c_dtype = self._as_complex_dtype(self.dtype)
        E = E.to(c_dtype)
        Eg = self.Eg.to(c_dtype)
        A = self.A.to(c_dtype)
        E0 = self.E0.to(c_dtype)
        Gamma = self.Gamma.to(c_dtype)

        # ---------- ε₂ (imaginary) ----------
        epsilon_i = torch.where(
            mask,
            (A * E0 * Gamma * (E - Eg) ** 2)
            / (E * ((E**2 - E0**2) ** 2 + Gamma**2 * E**2)),
            torch.zeros_like(E, dtype=c_dtype),
        )

        # ---------- ε₁ (real) helpers ----------
        a_ln = (Eg**2 - E0**2) * E**2 + Eg**2 * Gamma**2 - E0**2 * (E0**2 + 3 * Eg**2)
        a_atan = (E**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * Gamma**2
        a_alpha = torch.sqrt(4 * E0**2 - Gamma**2)
        a_gamma2 = E0**2 - 0.5 * Gamma**2
        a_ksi4 = (E**2 - a_gamma2) ** 2 + 0.25 * a_alpha**2 * Gamma**2

        # ---------- ε₁ contributions ----------
        εr1 = (
            (A * Gamma)
            / (torch.pi * a_ksi4)
            * (a_ln / (2 * a_alpha * E0))
            * torch.log((E0**2 + Eg**2 + a_alpha * Eg) / (E0**2 + Eg**2 - a_alpha * Eg))
        )

        εr2 = (
            (-A / (torch.pi * a_ksi4))
            * (a_atan / E0)
            * (
                torch.pi
                - torch.atan((a_alpha + 2 * Eg) / Gamma)
                + torch.atan((a_alpha - 2 * Eg) / Gamma)
            )
        )

        εr3 = (
            (2 * A * E0)
            / (torch.pi * a_ksi4 * a_alpha)
            * (Eg * (E**2 - a_gamma2))
            * (torch.pi + 2 * torch.atan(2 * (a_gamma2 - Eg**2) / (a_alpha * Gamma)))
        )

        εr4 = (
            (-A * E0 * Gamma)
            / (torch.pi * a_ksi4)
            * ((E**2 + Eg**2) / E)
            * torch.log(torch.abs(E - Eg) / (E + Eg))
        )

        εr5 = (
            (2 * A * E0 * Gamma * Eg)
            / (torch.pi * a_ksi4)
            * torch.log(
                torch.abs(E - Eg)
                * (E + Eg)
                / torch.sqrt((E0**2 - Eg**2) ** 2 + Eg**2 * Gamma**2)
            )
        )

        epsilon_r = εr1 + εr2 + εr3 + εr4 + εr5

        # ---------- clean NaNs / infs ----------
        epsilon_r = torch.nan_to_num(epsilon_r)
        epsilon_i = torch.nan_to_num(epsilon_i)

        return epsilon_r + 1j * epsilon_i

    def __repr__(self) -> str:
        """
        Return a string representation of the TaucLorentz dispersion instance.

        Returns:
            str: A string summarizing the TaucLorentz dispersion model with its parameters,
                 data type, and device.
        """
        return f"TaucLorentz Dispersion(Coefficients(Eg, A, E0, Gamma):{[self.Eg, self.A, self.E0, self.Gamma]})"
