"""
Authors:
    Sergei Rodionov, Daniele Veraldi
Date:
    2025-06-09
License:
    MIT, Open Source

================================================================================
Module: t_matrix.py
================================================================================
Description:
    This module implements the T_matrix class for computing the transfer matrix in 
    thin film optics using the Transfer Matrix Method (TMM). The T_matrix class 
    provides methods to calculate the transfer matrix for a coherent layer,incoherent layer,
    and interfaces as well as for propagation inside a layer. All layers are homogeneous and 
    isotropic. 

    All theoretical foundations and mathematical details of the implemented methods 
    are thoroughly discussed in the following references:

    - Mitsas, C. L., & Siapkas, D. I. (1995). Applied Optics, 34(10), 1678–1683. 

    - Troparevsky, M. C., et al. (2010). Optics Express, 18(24), 24715–24721. 

    - Byrnes, S. J. (2016). Multilayer optical calculations. arXiv preprint arXiv:1603.02720. 

    - Katsidis, C. C., & Siapkas, D. I. (2002). Applied Optics, 41(19), 3978–3987. 

    In the current implementation, each coherent layer is treated independently from the others 
    by introducing infinitesimally small air gaps between them. As a result, each layer is effectively 
    surrounded by air, allowing its T-matrix to be computed in isolation. The environmental (top) 
    and substrate (bottom) interfaces are added only at the final stage. While this approach simplifies 
    certain computations, it introduces unnecessary interface calculations that may lead to error accumulation. 
    **This design choice is subject to revision in future versions.**

    Matrix mutliplication procedure is implemented in model.py

    The implementation leverages a normalized wave-vector component (`nx`), 
    perpendicular to the layer boundaries, instead of directly using the angle of incidence. 
    This approach avoids dealing explicitly with complex-valued angles, 
    particularly beneficial for scenarios involving total internal reflection. 
    `nx` is computed once and consistently reused throughout the computation pipeline, 
    enhancing both numerical stability and computational efficiency.

    Additionally, PyTorch’s square root operation is defined such that the imaginary part of the result 
    is always non-negative, which helps avoid branch cut ambiguities.

    The methods in this module operate in a vectorized manner using 
    PyTorch tensors, which facilitates high-performance computations on both CPU 
    and GPU devices and compatible with automatic differentiation (autograd).

Key Components:
    - coherent_layer: Computes the transfer matrix for a single coherent 
      layer surrounded by air over all wavelengths and angles.
    - interface_s: Computes the interface matrix between two media for s-polarization.
    - interface_p: Computes the interface matrix between two media for p-polarization.
    - propagation_coherent: Computes the propagation transfer matrix through a layer.

Conventions:
    - propagation from left to right
    - refractive index defined as n_real + 1j*n_imm 
    - wavelenghts and thicknesses must be defined in nm
    - angles defined in degree in range [0, 90)

Usage:
    - in the case of very high absorption (alpha > 60) or total internal reflection 
    (nz purely imaginary) computational errors can arise. To avoid this we implement 
    the clamp-and-split trick threshold to avoid numerical instability. 


Example:
    >>> import torch
    >>> from t_matrix import T_matrix
    >>> tm = T_matrix(dtype=torch.complex64, device=torch.device('cpu'))
    >>> # Define optical parameters
    >>> n = torch.tensor([1 + 1.5j])
    >>> d = torch.tensor([100e-9])
    >>> wavelengths = torch.tensor([500e-9, 600e-9])
    >>> incidence_angle = torch.tensor([0,30,60])
    >>> nx = n * torch.sin(incidence_angle)
    >>> T = tm.coherent_layer('s', n, d, wavelengths, nx)
    >>> print(T)
================================================================================
"""


import torch
import numpy as np
from typing import List, Tuple
# Constants
c = 299792458  # Speed of light in vacuum (m/s)

class T_matrix:
    """
    Class to compute the transfer matrix.
    """
    def __init__(self, 
                 dtype: torch.dtype = torch.complex64,
                 device: torch.device = torch.device('cpu')) -> None:
        
        self.dtype = dtype
        self.device = device

    def coherent_layer(self,
                    pol: str, 
                    n: torch.Tensor, 
                    d: torch.Tensor, 
                    wavelengths: torch.Tensor, 
                    nx: torch.Tensor,
                    *,
                    clamp_alpha: float = 60.0,
                    ) -> torch.Tensor:
        """
        Computes the Transfer matrix for a single coherent layer surrounded by air 
        over all wavelengths and angles in parallel.  

        Parameters
        ----------
        n : torch.Tensor
            Refractive index of the layer. Shape: (num_wavelengths,)
        d : torch.Tensor
            Thickness of the layer. Must be broadcastable to n. 
        nx : torch.Tensor
            Transversal component of the k-vector normalized by k0. 
            Shape: (num_wavelengths, num_angles)
        Returns
        -------
        torch.Tensor
            Transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
        """
        n_air = torch.ones_like(n)
        if pol == 's':
            T_in = self.interface_s(n_air, n, nx)
            T_prop = self.propagation_coherent(n, d, wavelengths, nx, clamp_alpha = clamp_alpha)
            T_out = self.interface_s(n, n_air, nx)
            return T_in @ T_prop @ T_out
         
        elif pol == 'p':
            T_in = self.interface_p(n_air, n, nx)
            T_prop = self.propagation_coherent(n, d, wavelengths, nx, clamp_alpha = clamp_alpha)
            T_out = self.interface_p(n, n_air, nx)
            return T_in @ T_prop @ T_out
        
        else:
            raise ValueError(f"Invalid polarization: {pol}")
        

    def interface_s(self, 
                    ni: torch.Tensor, 
                    nf: torch.Tensor, 
                    nx: torch.Tensor) -> torch.Tensor:
        """
        Computes the boundary (interface) transfer matrix between two media for s-polarization,
        in parallel for all wavelengths and angles.

        Parameters
        ----------
        ni : torch.Tensor
            Refractive index of current layer. Shape: (num_wavelengths,)
        nf : torch.Tensor
            Refractive index of next layer. Same shape as ni
        nx : torch
            Transversal component of the k-vector normalized by k0. 
            Shape: (num_wavelengths, num_angles)

        Returns
        -------
        torch.Tensor
            Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
        # Getting propagation constants
        niz = torch.sqrt(ni[:,None]**2 - nx**2)
        nfz = torch.sqrt(nf[:,None]**2 - nx**2)

        T = torch.zeros(niz.shape + (2, 2), dtype=self.dtype, device=self.device)
        # Compute T matrix
        T[..., 0, 0] = 0.5*(1 + nfz / niz)
        T[..., 0, 1] = 0.5*(1 - nfz / niz)
        T[..., 1, 0] = 0.5*(1 - nfz / niz)
        T[..., 1, 1] = 0.5*(1 + nfz / niz)

        return T
    
    def interface_p(self, 
                    ni: torch.Tensor, 
                    nf: torch.Tensor, 
                    nx: torch.Tensor) -> torch.Tensor:
        """
        Computes the boundary (interface) transfer matrix between two media for p-polarization,
        in parallel for all wavelengths and angles.

        Parameters
        ----------
        ni : torch.Tensor
            Refractive index of current layer. Shape: (num_wavelengths,)
        nf : torch.Tensor
            Refractive index of next layer. Same shape as ni
        nx : torch
            Transversal component of the k-vector normalized by k0. 
            Shape: (num_wavelengths, num_angles)

        Returns
        -------
        torch.Tensor
            Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
        # Getting propagation constants
        niz = torch.sqrt(ni[:,None]**2 - nx**2)
        nfz = torch.sqrt(nf[:,None]**2 - nx**2)

        T = torch.zeros(niz.shape + (2, 2), dtype=self.dtype, device=self.device)
        coeff = (ni**2/nf**2)[:, None]
        # Compute T matrix
        T[..., 0, 0] = 0.5*(1 + coeff*nfz / niz)/torch.sqrt(coeff)
        T[..., 0, 1] = 0.5*(1 - coeff*nfz / niz)/torch.sqrt(coeff)
        T[..., 1, 0] = 0.5*(1 - coeff*nfz / niz)/torch.sqrt(coeff)
        T[..., 1, 1] = 0.5*(1 + coeff*nfz / niz)/torch.sqrt(coeff)

        return T
    
    def propagation_coherent(self, 
                    ni: torch.Tensor, 
                    d: torch.Tensor, 
                    wavelengths: torch.Tensor, 
                    nx: torch.Tensor,
                    *,
                    clamp_alpha: float = 60.0
                    ) -> torch.Tensor:
        """
        Computes the propagation transfer matrix for a layer,
        in parallel for all wavelengths and angles.

        Parameters
        ----------
        ni : torch.Tensor
            Refractive index of the layer. Shape: (num_wavelengths,)
        d : torch.Tensor
            Thickness of the layer. Must be broadcastable to n_i. 
        wavelengths : torch.Tensor
            Wavelength of light. Shape: (num_wavelengths,)
        nx : torch  
            Transversal component of the k-vector normalized by k0. 
            Shape: (num_wavelengths, num_angles)
        clamp_alpha: float
            Setup the clamp-and-split trick threshold to avoid numerical 
            instability, when dealing with high absorption. 

        Returns
        -------
        torch.Tensor
            Propagation matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
        # Getting propagation constant
        niz = torch.sqrt(ni[:, None]**2 - nx**2)           # complex kz / k0

        #Get delta
        k0     = 2 * torch.pi / wavelengths[:, None]       # real (λ,1)
        delta  = k0 * d * niz                              # complex (λ,θ)
        
        #Clamp-and-split trick to fix absorption anomalies
        beta   = delta.real                                # phase term  β
        alpha  = delta.imag.clamp(max=clamp_alpha)   # decay term  α ≥ 0 

        #Reassemble delta
        delta = beta + 1j*alpha

        #Get matrix components
        exp_fwd  = torch.exp(-1j*delta)                    # e^{α - iβ}
        exp_back = torch.exp(1j*delta)                       # e^{-α + iβ}

        # --- assemble 2×2 matrices ---------------------------------------------
        T = torch.zeros(delta.shape + (2, 2),
                        dtype=exp_fwd.dtype,
                        device=exp_fwd.device)
        T[..., 0, 0] = exp_fwd
        T[..., 1, 1] = exp_back
        return T
