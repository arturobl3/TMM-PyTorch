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
    
    @abstractmethod
    def __repr__(self) -> str:
        """method to return the dispersion parameters"""


class Constant_epsilon(BaseDispersion):
    """
    A dispersion model with a constant (flat) dielectric permittivity.
    This class implements a dispersion model in which the dielectric permittivity remains
    constant across all wavelengths. It inherits from the Dispersion base class.

    Attributes:
        epsilon_const (torch.nn.Parameter): The constant dielectric permittivity value.
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
            epsilon_const (torch.nn.Parameter): The constant dielectric permittivity value.
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
    
    def __repr__(self) -> str:
        """
        Return a string representation of the dispersion instance.

        Returns:
            str: A string summarizing the dispersion.
        """
        
        return (f"Constant Dispersion(epsilon:{self.epsilon_const}, dtype: {self.dtype}, device: {self.device})")



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
        A (torch.nn.Parameter): Oscillator amplitude.
        E0 (torch.nn.Parameter): Resonance energy.
        C (torch.nn.Parameter): Damping coefficient.
    """
    def __init__(self,
                 A: torch.nn.Parameter,
                 E0:torch.nn.Parameter,
                 C:torch.nn.Parameter,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
    )-> None:
        """
        Initialize the Lorentz dispersion model with given parameters.
        Args:
            A (torch.nn.Parameter): Oscillator amplitude.
            E0 (torch.nn.Parameter): Resonance energy.
            C (torch.nn.Parameter): Damping coefficient.
            dtype (torch.dtype): Data type for tensor computations.
            device (torch.device): Device (e.g., CPU or GPU) to use for tensor computations.
        """
        self.dtype = dtype
        self.device = device
        self.A = A
        self.E0 = E0
        self.C = C


    def refractive_index(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex refractive index at the given wavelengths
        The refractive index is calculated as the square root of the electric permittivity:
            n = sqrt(ε)

        Args:
            wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) at which to compute the refractive index.
        Returns:
            torch.Tensor: The computed complex refractive index.
        """
        return torch.sqrt(self.epsilon(wavelength))
    
    def epsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
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
        
        # Lorentz electric permittivity calculation
        epsilon = (self.A * self.E0) / (self.E0**2 - E**2 - 1j * self.C * E)
        
        return epsilon  
    
    def __repr__(self) -> str:
        """
        Return a string representation of the dispersion instance.

        Returns:
            str: A string summarizing the dispersion.
        """
        
        return (f"Lorentz Dispersion(coefficients (A,E0,C):{self.coefficients}, dtype: {self.dtype}, device: {self.device})")


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
        dtype (torch.dtype): Data type for tensor computations.
        device (torch.device): Device (e.g., CPU or GPU) on which computations are performed.
        A, B, C (torch.nn.Parameter): Coefficients for the real part of the refractive index.
        D, E, F (torch.nn.Parameter): Coefficients for the imaginary part (extinction) of the refractive index.
    """

    def __init__(self,
                 A = torch.nn.Parameter,
                 B = torch.nn.Parameter,
                 C = torch.nn.Parameter,
                 D = torch.nn.Parameter,
                 E = torch.nn.Parameter,
                 F = torch.nn.Parameter,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
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
            dtype (torch.dtype, optional): Data type for tensor operations. Defaults to torch.float.
            device (torch.device, optional): Device on which tensor operations will be executed. Defaults to CPU.
        """
        self.dtype = dtype
        self.device = device
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F

    def refractive_index(self, wavelength: torch.Tensor) -> torch.Tensor:
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
        
        n = self.A + 1e4 * self.B / wavelength**2 + 1e9 * self.C / wavelength**4
        k = self.D + 1e4 * self.E / wavelength**2 + 1e9 * self.F / wavelength**4
        return n + 1j * k

    def epsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
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
        return (self.refractive_index(wavelength))**2
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Cauchy dispersion instance.
        
        Returns:
            str: A string summarizing the Cauchy dispersion model with its coefficients,
                 data type, and device.
        """
        
        return (f"Cauchy Dispersion(Coefficients(A,B,C,D,E,F):{[self.A, self.B, self.C, self.D, self.E, self.F]}, dtype: {self.dtype}, device: {self.device})")


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
        dtype (torch.dtype): Data type for tensor computations.
        device (torch.device): Device on which tensor operations are performed.
        Eg (torch.nn.Parameter): Optical band gap energy.
        A (torch.nn.Parameter): Amplitude of the transition.
        E0 (torch.nn.Parameter): Resonance energy.
        C (torch.nn.Parameter): Broadening (damping) parameter.
    """

    def __init__(self,
                 Eg :torch.nn.Parameter,
                 A : torch.nn.Parameter,
                 E0 : torch.nn.Parameter,
                 C : torch.nn.Parameter,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),        
        ) -> None:
        """
        Initialize the TaucLorentz model with the specified parameters.
        
        Args:
            Eg (torch.nn.Parameter): Optical band gap energy.
            A (torch.nn.Parameter): Amplitude of the transition.
            E0 (torch.nn.Parameter): Resonance energy.
            C (torch.nn.Parameter): Broadening (damping) parameter.
            dtype (torch.dtype, optional): Data type for tensor operations. Defaults to torch.float.
            device (torch.device, optional): Device for tensor operations. Defaults to CPU.
        """
        self.dtype = dtype
        self.device = device
        self.Eg = Eg
        self.A = A
        self.E0 = E0
        self.C = C
    
    def refractive_index(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex refractive index at the given wavelengths.

        Args:
            wavelength (torch.Tensor): Tensor of wavelengths (in nanometers) for evaluation.
        Returns:
            torch.Tensor: Complex refractive index evaluated at the specified wavelengths.
        """
        return torch.sqrt(self.epsilon(wavelength))
    
    def epsilon(self, wavelength: torch.Tensor):
        """
        Compute the complex dielectric function using the Tauc-Lorentz model.

        The photon energy E is calculated from the wavelength (meters) using:
            E = (h * c / e) / (wavelength)
        where:
            - h (Planck constant) = 6.62607015e-34 J·s,
            - c (speed of light) = 299792458 m/s,
            - e (elementary charge) = 1.60217663e-19 C.

        The model parameters are unpacked as:
            Eg: Optical band gap energy.
            A: Amplitude of the transition.
            E0: Resonance energy.
            C: Broadening (damping) parameter.

        For photon energies E greater than Eg, the imaginary part ε_i is computed by:
            ε_i = (1/E) * (A * E0 * C * (E - Eg)^2) / ((E^2 - E0^2)^2 + C^2 * E^2)
        For E ≤ Eg, ε_i is set to 0.

        The real part ε_r is obtained from several contributions (epsilon_r1 through epsilon_r5)
        that involve logarithmic and arctan functions to satisfy the Kramers-Kronig relations.

        Finally, any NaN values in ε_r and ε_i are replaced by zero, and the complex dielectric function is:
            ε = ε_r + i * ε_i

        Args:
            wavelength (torch.Tensor): Tensor of wavelengths (meters) for evaluation.
        Returns:
            torch.Tensor: Complex dielectric function evaluated at the specified wavelengths.
        """

        #Constants
        plank_constant = 6.62607015*1e-34
        c_constant = 299792458
        e_constant = 1.60217663*1e-19
        
        E = (plank_constant*c_constant/e_constant)/(wavelength)

        
        #Calculation of imaginary part of electric permittivity
        if E > self.Eg:
            epsilon_i = (1/E)*(self.A*self.E0*self.C*(E - self.Eg)**2)/((E**2 - self.E0**2)**2 + self.C**2*E**2)
        else:
            epsilon_i = 0

        #Calculation of real part of electric permittivity
        a_ln = (self.Eg**2 - self.E0**2)*(E**2) + (self.Eg**2)*(self.C**2) - (self.E0**2)*(self.E0**2 + 3*self.Eg**2)
        a_atan = (E**2 - self.E0**2)*(self.E0**2 + self.Eg**2) + self.Eg**2*self.C**2
        a_alpha = torch.sqrt(4*self.E0**2 - self.C**2)
        a_gamma2 = (self.E0**2 - 0.5 * (self.C**2))
        a_ksi4 = (E**2 - a_gamma2)**2 + 0.25 * a_alpha**2 * self.C**2
        
        epsilon_r1_1 = (self.A * self.C)/(torch.pi * a_ksi4)
        epsilon_r1_2 = a_ln / (2 * a_alpha * self.E0)
        epsilon_r1_3 = self.E0**2 + self.Eg**2 + a_alpha * self.Eg
        epsilon_r1_4 = self.E0**2 + self.Eg**2 - a_alpha * self.Eg
        epsilon_r1_5 = torch.log(epsilon_r1_3 / epsilon_r1_4)
        
        epsilon_r1 = epsilon_r1_1 * epsilon_r1_2 * epsilon_r1_5
        
        epsilon_r2_1 = - self.A / (torch.pi * a_ksi4)
        epsilon_r2_2 = a_atan / self.E0
        epsilon_r2_3 = (a_alpha + 2 * self.Eg) / self.C
        epsilon_r2_4 = (a_alpha - 2 * self.Eg) / self.C
        epsilon_r2_5 = torch.pi - torch.arctan(epsilon_r2_3) + torch.arctan(epsilon_r2_4)
    
        epsilon_r2 = epsilon_r2_1 * epsilon_r2_2 * epsilon_r2_5
        
        epsilon_r3_1 = (2 * self.A * self.E0)/(torch.pi * a_ksi4 * a_alpha)
        epsilon_r3_2 = self.Eg * (E**2 - a_gamma2)
        epsilon_r3_3 = 2 * (a_gamma2 - self.Eg**2) / (a_alpha * self.C)
        epsilon_r3_4 = torch.pi + 2 * torch.arctan(epsilon_r3_3)
        
        epsilon_r3 = epsilon_r3_1 * epsilon_r3_2 * epsilon_r3_4
        
        epsilon_r4_1 = -(self.A * self.E0 * self.C) / (torch.pi * a_ksi4)
        epsilon_r4_2 = (E**2 + self.Eg**2) / E
        epsilon_r4_3 = torch.log(torch.abs(E - self.Eg) / (E + self.Eg))
        epsilon_r4 = epsilon_r4_1 * epsilon_r4_2 * epsilon_r4_3
        
        epsilon_r5_1 = (2 * self.A * self.E0 * self.C * self.Eg) / (torch.pi * a_ksi4)
        epsilon_r5_2 = torch.abs(E - self.Eg) * (E + self.Eg)
        epsilon_r5_3 = torch.sqrt((self.E0**2 - self.Eg**2)**2 + self.Eg**2 * self.C**2)
        epsilon_r5_4 = torch.log(epsilon_r5_2 / epsilon_r5_3)
        epsilon_r5 = epsilon_r5_1 * epsilon_r5_4
        
        khi_r = epsilon_r1 + epsilon_r2 + epsilon_r3 + epsilon_r4 + epsilon_r5
        epsilon_r = khi_r

        #Prevent NaN values apperance
        epsilon_r[torch.isnan(epsilon_r)] = 0
        epsilon_i[torch.isnan(epsilon_i)] = 0
        
        #Refractive index calculation
        eps = epsilon_r + 1j * epsilon_i

        return eps
    
    def __repr__(self) -> str:
        """
        Return a string representation of the TaucLorentz dispersion instance.
        
        Returns:
            str: A string summarizing the TaucLorentz dispersion model with its parameters,
                 data type, and device.
        """
        return (f"TaucLorentz Dispersion(Coefficients(Eg, A, E0, C):{[self.Eg, self.A, self.E0, self.C]}, dtype: {self.dtype}, device: {self.device})")

