from typing import List, Tuple, Literal
from T_matrix import T_matrix
import torch 

class OpticalProperties:
    """
    Encapsulates the computed optical properties of a multilayer optical system.

    This class stores the transfer matrices for s- and p-polarizations, along with the
    refractive indices of the environment and substrate, and the x-component of the wavevector.
    It provides methods to compute the reflectance and transmittance of the system based on
    these parameters.

    Attributes:
        Tm_s (T_matrix): Transfer matrix for s-polarization.
        Tm_p (T_matrix): Transfer matrix for p-polarization.
        n_env (torch.Tensor): Refractive index of the environment (incident medium).
        n_subs (torch.Tensor): Refractive index of the substrate (transmission medium).
        nx (torch.Tensor): x-component of the wavevector, typically computed as n_env * sin(angle).
    """
    def __init__(self,
                Tm_s: T_matrix,
                Tm_p: T_matrix,
                n_env:torch.Tensor,
                n_subs: torch.Tensor,
                nx:torch.Tensor    
    ) -> None:
        """
        Initialize the OpticalProperties instance with transfer matrices and refractive indices.

        Args:
            Tm_s (T_matrix): Transfer matrix for s-polarization.
            Tm_p (T_matrix): Transfer matrix for p-polarization.
            n_env (torch.Tensor): Tensor of refractive indices for the environment.
            n_subs (torch.Tensor): Tensor of refractive indices for the substrate.
            nx (torch.Tensor): Tensor representing the x-component of the wavevector.
        """
        self.Tm_s = Tm_s
        self.Tm_p = Tm_p
        self.n_env = n_env
        self.n_subs = n_subs
        self.nx = nx
        
    def reflection(self):
        """
        Calculate the reflectance for s- and p-polarizations.

        The reflection coefficient for each polarization is computed from the corresponding
        transfer matrix T using:
            r = T[1, 0] / T[0, 0]
        The reflectance is then obtained by taking the squared magnitude of the reflection coefficient.

        Returns:
            tuple(torch.Tensor, torch.Tensor): A tuple containing:
                - Reflectance for s-polarization.
                - Reflectance for p-polarization.
        """
      
        r_s = self.Tm_s[:, :, 1, 0]/self.Tm_s[:, :, 0, 0]
        r_p = self.Tm_p[:, :, 1, 0]/self.Tm_p[:, :, 0, 0]

        return torch.abs(r_s)**2, torch.abs(r_p)**2
        

    def transmission(self):
        """
        Calculate the transmittance for s- and p-polarizations.

        The method first computes the normal (z) components of the refractive indices for the
        environment and substrate using:
            n1z = sqrt(n_env^2 - nx^2) and n2z = sqrt(n_subs^2 - nx^2)
        Then, the transmission coefficient is approximated as:
            t = 1 / T[0, 0]
        and the transmittance is given by:
            T = |t|^2 * Re(n2z / n1z)

        Returns:
            tuple(torch.Tensor, torch.Tensor): A tuple containing:
                - Transmittance for s-polarization.
                - Transmittance for p-polarization.
        """

        n1z = torch.sqrt(self.n_env[:,None]**2 - self.nx**2)
        n2z = torch.sqrt(self.n_subs[:,None]**2 - self.nx**2)

        t_s = 1/self.Tm_s[:, :, 0, 0] 
        t_p = 1/self.Tm_p[:, :, 0, 0] 

        T_s = torch.abs(t_s)**2 * torch.real(n2z/n1z)
        T_p = torch.abs(t_p)**2 * torch.real(n2z/n1z)

        return T_s, T_p


    