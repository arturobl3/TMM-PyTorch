from typing import List, Tuple
from abc import ABC, abstractmethod
import torch 

class BaseDispersion(ABC):
    """ Abstract base class to define dispersion models for materials"""

    @abstractmethod
    def epsilon(self, wavelengths: torch.Tensor, *args, **kwargs) -> torch.tensor:
        """method to calculate the dielectric constant of a material"""

    @abstractmethod
    def refractive_index(self, wavelengths: torch.Tensor, *args, **kwargs) -> torch.tensor:
        """method to calculate the refractive index of a material"""


class Constant_epsilon(BaseDispersion):
    """
    A dispersion model with a constant (flat) dielectric permittivity.
    This class implements a dispersion model in which the dielectric permittivity remains
    constant across all wavelengths. It inherits from the Dispersion base class.

    Attributes:
        epsilon_const (float | complex): The constant dielectric permittivity value.
        dtype (torch.dtype): The data type for the torch tensor (e.g., torch.float32).
        device (torch.device): The device on which to allocate tensors (e.g., CPU or GPU).
    """

    def __init__(self,
                 epsilon_const: torch.nn.Parameter,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
    )-> None:
        """
        Initialize the flatRefractiveIndex instance.

        Args:
            epsilon_const (float | complex): The constant dielectric permittivity value.
            dtype (torch.dtype): The desired data type for the output tensors.
            device (torch.device): The device on which the tensors should be allocated.
        """
        self.epsilon_const = epsilon_const
        self.dtype = dtype
        self.device = device


    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the dielectric permittivity (epsilon).
        The dielectric permittivity is calculated as the square of the refractive index.
        Returns:
            torch.tensor: A 1D tensor of size `wavelengths` where each element is set
                          to the constant refractive index value `n`.
        """
        epsilon = self.epsilon(wavelengths)
        return torch.sqrt(epsilon)
    
    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Generate the dielectric permittivity tensor.
        Returns:
            torch.tensor: A tensor representing the dielectric permittivity across the wavelengths.
        """
        return self.epsilon_const * torch.ones_like(wavelengths, dtype=self.dtype, device= self.device)
    
      


class Lorentz(BaseDispersion):  
    """
    Implements the Lorentz oscillator model for optical dispersion.
    This class computes the electric permittivity and refractive index based on the Lorentz oscillator model.
    It extends the BaseDispersion class and uses PyTorch tensors for numerical computations,
    allowing for efficient evaluation on both CPU and GPU devices.
    
    Attributes:
        dtype (torch.dtype): The data type used for tensor computations.
        device (torch.device): The device (e.g., CPU or GPU) on which the tensors are allocated.
        wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) at which the dispersion properties are evaluated.
        coefficients (list of torch.Tensor): A list containing the Lorentz oscillator parameters:
            - A (torch.Tensor): Oscillator amplitude.
            - E0 (torch.Tensor): Resonance energy.
            - C (torch.Tensor): Damping coefficient.
    """
    def __init__(self,
                 A: float,
                 E0:float,
                 C:float,
                 wavelength : torch.Tensor,
                 dtype: torch.dtype,
                 device: torch.device,
    )-> None:
        """
        Initialize the Lorentz dispersion model with given parameters.
        Args:
            A (float): Oscillator amplitude.
            E0 (float): Resonance energy.
            C (float): Damping coefficient.
            wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) for dispersion evaluation.
            dtype (torch.dtype): Data type for tensor computations.
            device (torch.device): Device (e.g., CPU or GPU) to use for tensor computations.
        """
        self.dtype = dtype
        self.device = device
        self.wavelength = wavelength

        self.coefficients = [torch.tensor(A, dtype=self.dtype, device=self.device),
                             torch.tensor(E0, dtype=self.dtype, device=self.device),
                             torch.tensor(C, dtype=self.dtype, device=self.device)]


    def getRefractiveIndex(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex refractive index at the given wavelengths
        The refractive index is calculated as the square root of the electric permittivity:
            n = sqrt(ε)

        Args:
            wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) at which to compute the refractive index.
        Returns:
            torch.Tensor: The computed complex refractive index.
        """
        return torch.sqrt(self.getEpsilon(wavelength))
    
    def getEpsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex electric permittivity using the Lorentz oscillator model.
        The electric permittivity ε is computed using the formula:
            ε = (A * E0) / (E0^2 - E^2 - i * C * E)
        where E is the photon energy calculated as:
            E = (h * c / e) / (wavelength)  
        Constants:
            - h (Planck constant): 6.62607015e-34 J·s
            - c (Speed of light): 299792458 m/s
            - e (Elementary charge): 1.60217663e-19 C 
        
        Args:
            wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) at which to compute the permittivity.
        Returns:
            torch.Tensor: The computed complex electric permittivity.
        """
        # Constants
        plank_constant = torch.tensor(6.62607015e-34, dtype=self.dtype, device = self.device)
        c_constant = torch.tensor(299792458, dtype=self.dtype, device = self.device)
        e_constant = torch.tensor(1.60217663e-19, dtype=self.dtype, device = self.device)
        
        E = (plank_constant * c_constant / e_constant) / (wavelength)

        A, E0, C = self.coefficients
        
        # Lorentz electric permittivity calculation
        e = (A * E0) / (E0**2 - E**2 - 1j * C * E)
        
        return e
    
    def updateParams(self, new_A: float, new_E0:float, new_C:float) -> None:
        self.coefficients = [torch.tensor(new_A, dtype=self.dtype, device=self.device),
                             torch.tensor(new_E0, dtype=self.dtype, device=self.device),
                             torch.tensor(new_C, dtype=self.dtype, device=self.device)]
