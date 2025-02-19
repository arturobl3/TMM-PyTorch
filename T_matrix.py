"""
TMM-fjfjhfjhf
Description: 
Authors: 
Date:
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
                    nx: torch.Tensor) -> torch.Tensor:
        """
        Computes the total transfer matrix for a single coherent layer surrounded by air 
        over all wavelengths and angles in parallel.

        Parameters
        ----------
        n : torch.Tensor
            Refractive index of the layer. Shape: (num_wavelengths,)
        d : torch.Tensor
            Thickness of the layer. Must be broadcastable to n_i. 
        nx : torch.Tensor
            Transversal component of the k-vector normalized by k0. Shape: (num_wavelengths, num_angles)
        Returns
        -------
        torch.Tensor
            Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
        """
        n_air = torch.ones_like(n)
        if pol == 's':
            T_in = self.interface_s(n_air, n, nx)
            T_prop = self.propagation_coherent(n, d, wavelengths, nx)
            T_out = self.interface_s(n, n_air, nx)
            return torch.einsum('...ij,...jk->...ik', T_in, torch.einsum('...ij,...jk->...ik', T_prop, T_out ))
         
        elif pol == 'p':
            T_in = self.interface_p(n_air, n, nx)
            T_prop = self.propagation_coherent(n, d, wavelengths, nx)
            T_out = self.interface_p(n, n_air, nx)
            return torch.einsum('...ij,...jk->...ik', T_in, torch.einsum('...ij,...jk->...ik', T_prop, T_out ))
        
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
            Transversal component of the k-vector normalized by k0. Shape: (num_wavelengths, num_angles)

        Returns
        -------
        torch.Tensor
            Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
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
            Transversal component of the k-vector normalized by k0. Shape: (num_wavelengths, num_angles)

        Returns
        -------
        torch.Tensor
            Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
        niz = torch.sqrt(ni[:,None]**2 - nx**2)
        nfz = torch.sqrt(nf[:,None]**2 - nx**2)

        T = torch.zeros(niz.shape + (2, 2), dtype=self.dtype, device=self.device)
        coeff = (ni**2/nf**2)[:, None]
        # Compute T matrix
        T[..., 0, 0] = 0.5*(1 + coeff*nfz / niz)
        T[..., 0, 1] = 0.5*(1 - coeff*nfz / niz)
        T[..., 1, 0] = 0.5*(1 - coeff*nfz / niz)
        T[..., 1, 1] = 0.5*(1 + coeff*nfz / niz)

        return T
    
    def propagation_coherent(self, 
                    ni: torch.Tensor, 
                    d: torch.Tensor, 
                    wavelengths: torch.Tensor, 
                    nx: torch.Tensor) -> torch.Tensor:
        """
        Computes the propagation transfer matrix for through a layer,
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
            Transversal component of the k-vector normalized by k0. Shape: (num_wavelengths, num_angles)

        Returns
        -------
        torch.Tensor
            Propagation matrices of shape (num_wavelengths, num_angles, 2, 2)
        """
        niz = torch.sqrt(ni[:,None]**2 - nx**2)
        delta_i = (2 * np.pi / wavelengths[:,None]) * niz * d

        T = torch.zeros(delta_i.shape + (2, 2), dtype=self.dtype, device=self.device)
        T[..., 0, 0] = torch.exp(-1j*delta_i)
        T[..., 1, 1] = torch.exp(1j*delta_i)

        return T