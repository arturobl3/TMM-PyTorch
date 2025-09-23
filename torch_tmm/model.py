"""
Authors:
    Sergei Rodionov, Daniele Veraldi
Date:
    2025-06-09
License:
    MIT, Open Source

================================================================================
Module: model.py
================================================================================
Description:
    This module defines the optical model used for simulation. It collects all layers,
    wavelengths, and angles, and computes the corresponding T-matrices for both in-plane (p-polarized)
    and out-of-plane (s-polarized) light. The result is an instance of an optical calculator that enables
    the evaluation of various optical quantities (transmission, reflection, ...).

Key Components:
    - `Model`: main simulation object encapsulating structure, environment, and substrate.
    - `forward`: computes T-matrices across input wavelengths and angles.
    - `T_matrix`: utility class that computes interface and layer matrices.
    - `OpticalCalculator`: wrapper for downstream optical computations.

Conventions:
    - Optical wave propagates left to right (from environment to substrate).
    - Wavelengths and layer thicknesses must be specified in nanometers (nm).
    - Incident angles are in degrees and must be in the range [0°, 90°).

Example:
    >>> import torch
    >>> from torch_tmm import Dispersion, Material, Model, Layer
    >>> #Define dtype and device
    >>> dtype = torch.float32
    >>> device = torch.device('cpu')
    >>> #Define wavelengths (nm) and angles (deg)
    >>> wavelengths = torch.linspace(400, 800, 801)
    >>> angles = torch.linspace(0, 89, 357)
    >>> #Define materials
    >>> subs_mat = Material([Dispersion.Constant_epsilon(1)], name='Air')
    >>> env_mat = Material([Dispersion.Constant_epsilon(1.46**2)], name='Fused-silica')
    >>> metal_mat = Material([Dispersion.Lorentz(A=80, E0=0.845, Gamma=0.1),
                        Dispersion.Constant_epsilon(1)], name='Metal')
    >>> #Define layers
    >>> env = Layer(env_mat, layer_type='semi-inf')
    >>> subs = Layer(subs_mat, layer_type='semi-inf')
    >>> metal = Layer(metal_mat, layer_type='coh', thickness=25) #nm
    >>> #Define model
    >>> optical_model = Model(env=env, structure=[metal], subs=subs,
                                    dtype=dtype, device=device)
    >>> calc = optical_model(wavelengths, angles)
    >>> Rs = calc.reflection('s')
    >>> Rp = calc.reflection('p')
    >>> Ts = calc.transmission('s')
    >>> Tp = calc.transmission('p')
================================================================================
"""

from .layer import BaseLayer
from .t_matrix import T_matrix
from .optical_calculator import OpticalCalculator
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Transfer-matrix model of a multilayer stack.

    The model owns exactly **one** environment layer (``layer_type='env'``),
    **one** substrate layer (``'subs'``) and an ordered
    :class:`~torch.nn.ModuleList` of intermediate layers
    (typically ``'coh'``).  All children move automatically with
    :py:meth:`~torch.nn.Module.to`.

    Parameters
    ----------
    env : BaseLayer
        Incident medium (must have ``layer_type='env'``).
    structure : list[BaseLayer]
        Finite stack between env and substrate (must *not* contain
        ``'env'`` or ``'subs'`` layers).
    subs : BaseLayer
        Transmission medium (must have ``layer_type='subs'``).
    dtype : torch.dtype, default ``torch.float32``
        Either ``float32`` or ``float64``; determines the complex
        precision internally (``complex64`` / ``complex128``).
    device : torch.device, default ``cpu``
        Where all tensors live initially.
    """

    def __init__(
        self,
        env: BaseLayer,
        structure: list[BaseLayer],
        subs: BaseLayer,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        # --------------------- validate dtype --------------------------------
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"dtype must be float32 or float64, got {dtype!s}")

        # --------------------- validate layer roles --------------------------
        if env.layer_type != "semi-inf":
            raise ValueError("`env` layer must have layer_type='semi-inf'.")
        if subs.layer_type != "semi-inf":
            raise ValueError("`subs` layer must have layer_type='semi-inf'.")
        for i, lyr in enumerate(structure, 1):
            if lyr.layer_type in ("semi-inf"):
                raise ValueError(
                    f"structure[{i}] has invalid layer_type={lyr.layer_type!r}"
                )

        # --------------------- register children -----------------------------
        self.env: BaseLayer = env
        self.subs: BaseLayer = subs
        self.structure = nn.ModuleList(structure)  # keeps order & registers

        # --------------------- mirrors & helper objects ----------------------
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device
        self._c_dtype: torch.dtype = (
            torch.complex64 if dtype == torch.float32 else torch.complex128
        )
        self.T_matrix = T_matrix(self._c_dtype, self._device)

        # move everything to the requested dtype/device once
        self._smart_to(dtype=dtype, device=device)

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
            nn.Module.to(self, device=device)

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

    # ----------------------------------------------------------------- API --
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    # ---------------------------------------------------------------- .to() --
    def _sync_dtype_device(self) -> None:
        """Refresh mirrors from the first parameter/buffer."""
        try:
            p = next(self.parameters())
        except StopIteration:
            try:
                p = next(self.buffers())
            except StopIteration:
                return
        self._dtype, self._device = p.dtype, p.device
        self._c_dtype = (
            torch.complex64 if self._dtype == torch.float32 else torch.complex128
        )
        # re-instantiate helper that caches dtype/device
        self.T_matrix = T_matrix(self._c_dtype, self._device)

    def _apply(self, fn, recurse=True):
        out = nn.Module._apply(self, fn, recurse)  # moves all parameters & buffers
        self._sync_dtype_device()
        return out

    def __repr__(self) -> str:
        """
        Return a string representation of the Model instance.

        The representation includes the types of the environment and substrate layers,
        the number of layers in the structure, and the data type and device used for computations.

        Returns:
            str: A string summarizing the Model.
        """
        env_repr = repr(self.env)
        subs_repr = repr(self.subs)
        structure_repr = f"[{', '.join(repr(layer) for layer in self.structure)}]"
        return (
            f"Model(\n"
            f"  Environment: {env_repr},\n"
            f"  Structure: {structure_repr} (n={len(self.structure)} layers),\n"
            f"  Substrate: {subs_repr},\n"
            f"  Dtype: {self.dtype}, Device: {self.device}\n"
            f")"
        )

    # --------------------------------------------------------------------- forward
    def forward(
        self,
        wavelengths: torch.Tensor,  # 1-D  (L,)
        angles: torch.Tensor,  # 1-D  (A,) in degrees
    ) -> OpticalCalculator:
        """
        Compute optical response for every wavelength / angle pair.

        Parameters:
            wavelengths : torch.Tensor
            Wavelengths in **nanometres** (positive, floating tensor).

            angles : torch.Tensor
            Angles in **degreed** (positive, floating tensor).

        Returns
        -------
        OpticalCalculator
            Holds T-matrices for s & p polarisations and auxiliary indices.
        """
        # ---------- normalise inputs ----------------------------------------
        if wavelengths.ndim != 1 or angles.ndim != 1:
            raise ValueError("wavelengths and angles must be 1-D tensors.")

        wl = wavelengths.to(dtype=self.dtype, device=self.device)  # (L,)
        th = torch.deg2rad(angles).to(dtype=self.dtype, device=self.device)  # (A,)

        # ---------- refractive indices --------------------------------------
        n_env = self.env.refractive_index(wl)  # (L,)
        n_subs = self.subs.refractive_index(wl)  # (L,)
        n_air = torch.ones_like(n_env, dtype=self._c_dtype)

        nx = n_env[:, None] * torch.sin(th)[None, :]  # (L,A)

        # ---------- s-polarisation ------------------------------------------
        T_s = self._stack_transfer(
            wl, th, nx, pol="s", n_air=n_air, n_env=n_env, n_subs=n_subs
        )

        # ---------- p-polarisation ------------------------------------------
        T_p = self._stack_transfer(
            wl, th, nx, pol="p", n_air=n_air, n_env=n_env, n_subs=n_subs
        )

        return OpticalCalculator(Tm_s=T_s, Tm_p=T_p, n_env=n_env, n_subs=n_subs, nx=nx)

    # ----------------------------------------------------------------  helpers
    def _stack_transfer(
        self,
        wl: torch.Tensor,
        th: torch.Tensor,
        nx: torch.Tensor,
        *,
        pol: str,
        n_air: torch.Tensor,
        n_env: torch.Tensor,
        n_subs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build full 2×2 transfer matrix for one polarisation.

        Returns ``T[L,A,2,2]``.
        """
        # Interfaces: env ↔ air  (air is modelling the stack entrance)
        T_env = (
            self.T_matrix.interface_s if pol == "s" else self.T_matrix.interface_p
        )(
            n_env, n_air, nx
        )  # (L,A,2,2)

        # Internal stack
        T_stack = self._structure_matrix(wl, nx, pol)

        # Exit interface: air ↔ substrate
        T_subs = (
            self.T_matrix.interface_s if pol == "s" else self.T_matrix.interface_p
        )(
            n_air, n_subs, nx
        )  # (L,A,2,2)

        # total = T_env · T_stack · T_subs
        return T_env @ T_stack @ T_subs  # (L,A,2,2)

    # ------------------------------------------------------------ structure mat
    def _structure_matrix(
        self,
        wl: torch.Tensor,  # (L,)
        nx: torch.Tensor,  # (L,A)
        pol: str,
    ) -> torch.Tensor:
        """
        Multiply coherent layers from top (env side) to bottom (subs).

        Returns ``T[L,A,2,2]`` (identity if stack is empty).
        """
        L, A = wl.shape[0], nx.shape[1]
        T_tot = torch.eye(2, dtype=self._c_dtype, device=self.device).expand(
            L, A, 2, 2
        )  # broadcasted identity

        for layer in self.structure:  # only 'coh' layers
            d = layer.thickness  # scalar parameter
            n_l = layer.refractive_index(wl)  # (L,)

            # layer transfer matrix  → (L,A,2,2)
            T_l = self.T_matrix.coherent_layer(
                pol=pol, n=n_l, d=d, wavelengths=wl, nx=nx
            )

            T_tot = T_tot @ T_l  # batched matmul

        return T_tot
