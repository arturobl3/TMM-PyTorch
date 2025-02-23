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
    
    def updateParams(self, new_coefficients: List[float]) -> None:
        self.coefficients =  [torch.tensor(coeff, dtype=self.dtype, device=self.device) for coeff in new_coefficients]
        

class Cauchy(BaseDispersion):
    """
    Implements the Cauchy dispersion model for optical materials.
    
    This model expresses the complex refractive index as a function of wavelength
    using the Cauchy equations for both the real and imaginary parts. The model
    uses six coefficients which are scaled appropriately in the formulas.
    
    Attributes:
        dtype (torch.dtype): Data type for tensor computations.
        device (torch.device): Device (e.g., CPU or GPU) on which computations are performed.
        wavelength (torch.Tensor): Tensor of wavelengths at which the dispersion is evaluated.
        coefficients (List[torch.Tensor]): List of six coefficients [A, B, C, D, E, F] converted to tensors.
    """

    def __init__(self,
                 coefficients: List[float],
                 wavelength: torch.Tensor,
                 dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Initialize the Cauchy dispersion model with specified coefficients and parameters.
        
        Args:
            coefficients (List[float]): A list of six floats representing the Cauchy coefficients:
                A, B, C for the real part and D, E, F for the imaginary part.
            wavelength (torch.Tensor): Tensor of wavelengths at which to evaluate the model.
            dtype (torch.dtype): Data type to be used for all tensor operations.
            device (torch.device): Device on which tensor operations will be executed.
        """
        self.dtype = dtype
        self.device = device
        self.wavelength = wavelength
        # Convert each coefficient to a torch tensor with the specified dtype and device.
        self.coefficients = [torch.tensor(coeff, dtype=self.dtype, device=self.device) for coeff in coefficients]

    def getRefractiveIndex(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Calculate the complex refractive index at the given wavelengths using the Cauchy model.
        The real part n and the imaginary part k are computed as:
            n = A + (1e4 * B) / wavelength² + (1e9 * C) / wavelength⁴
            k = D + (1e4 * E) / wavelength² + (1e9 * F) / wavelength⁴
        The complex refractive index is then n + 1j*k.

        Args:
            wavelength (torch.Tensor): Tensor of wavelengths at which to compute the refractive index.
        Returns:
            torch.Tensor: Complex refractive index evaluated at the specified wavelengths.
        """
        A, B, C, D, E, F = self.coefficients
        n = A + 1e4 * B / wavelength**2 + 1e9 * C / wavelength**4
        k = D + 1e4 * E / wavelength**2 + 1e9 * F / wavelength**4
        return n + 1j * k

    def getEpsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex electric permittivity (dielectric constant) at the specified wavelengths.
        The permittivity is obtained by squaring the complex refractive index
        
        Args:
            wavelength (torch.Tensor): Tensor of wavelengths at which to compute the permittivity.
        Returns:
            torch.Tensor: The complex electric permittivity evaluated at the specified wavelengths.
        """
        # Here, self.refractive_index is assumed to be defined in BaseDispersion or elsewhere.
        # If not, consider replacing it with self.getRefractiveIndex.
        return (self.getRefractiveIndex(wavelength))**2

    def updateParams(self, new_coefficients: List[float]) -> None:
        """
        Update the Cauchy coefficients for the dispersion model.
        This method replaces the current coefficients with a new set provided as a list of floats.
        Each new coefficient is converted to a torch tensor with the appropriate dtype and device.
        
        Args:
            new_coefficients (List[float]): A list of six new coefficient values to update the model.
        """
        self.coefficients = [torch.tensor(coeff, dtype=self.dtype, device=self.device) for coeff in new_coefficients]
