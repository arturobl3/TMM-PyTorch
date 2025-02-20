from typing import List, Tuple
from abc import ABC, abstractmethod
import torch 

class BaseDispersion(ABC):
    """ Abstract base class to define dispersion models for materials"""

    @abstractmethod
    def getEpsilon(self, *args, **kwargs)->torch.tensor:
        """method to calculate the dielectric constant of a material"""

    @abstractmethod
    def getRefractiveIndex(self, *args, **kwargs)->torch.tensor:
        """method to calculate the refractive index of a material"""

    @abstractmethod
    def updateParams(self, *args, **kwargs)->None:
        """method used to update the model parameters for optimization"""


class flatEpsilon(BaseDispersion):
    """
    A dispersion model with a constant (flat) dielectric permittivity.
    This class implements a dispersion model in which the dielectric permittivity remains
    constant across all wavelengths. It inherits from the Dispersion base class.

    Attributes:
        e (float | complex): The constant dielectric permittivity value.
        num_wavelenght (int): The number of wavelength points to be generated.
        dtype (torch.dtype): The data type for the torch tensor (e.g., torch.float32).
        device (torch.device): The device on which to allocate tensors (e.g., CPU or GPU).
    """

    def __init__(self,
                 epsilon: torch.Tensor,
                 num_wavelenght : int,
                 dtype: torch.dtype,
                 device: torch.device,
    )-> None:
        """
        Initialize the flatRefractiveIndex instance.

        Args:
            epsilon (float | complex): The constant dielectric permittivity value.
            num_wavelenght (int): The number of wavelength points to generate.
            dtype (torch.dtype): The desired data type for the output tensors.
            device (torch.device): The device on which the tensors should be allocated.
        """
        self.e = epsilon
        self.num_wavelenght = num_wavelenght
        self.dtype = dtype
        self.device = device


    def getRefractiveIndex(self) -> torch.Tensor:
        """
        Compute the dielectric permittivity (epsilon).
        The dielectric permittivity is calculated as the square of the refractive index.
        
        Returns:
            torch.tensor: A 1D tensor of size `num_wavelenght` where each element is set
                          to the constant refractive index value `n`.
        """
        e = self.getEpsilon()
        return torch.sqrt(e)
    
    def getEpsilon(self) -> torch.Tensor:
        """
        Generate the dielectric permittivity tensor.
        
        Returns:
            torch.tensor: A tensor representing the dielectric permittivity across the wavelengths.
        """
        return torch.ones(self.num_wavelenght, dtype=self.dtype, device= self.device) * self.e
    
    def updateParams(self, new_epsilon: float | complex) -> None:
        """
        Update the dielectric permittivity parameter.

        Args:
            new_refractiveIndex (float | complex): The new value for the constant dielectric permittivity.
        """
        self.e = new_epsilon  
