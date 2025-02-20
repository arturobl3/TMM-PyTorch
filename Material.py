import torch
from typing import List
from Dispersion import BaseDispersion

class BaseMaterial():
    """
    BaseMaterial aggregates multiple dispersion models to represent an optical material.

    This class is designed to combine the contributions of several dispersion models 
    to compute the overall refractive index of a material. It is particularly useful 
    in optical simulations and thin film optimization where the optical response 
    may result from several dispersion effects.

    Attributes:
        dispersion (List[Dispersion]): A list of dispersion model instances. Each instance
            must implement a `getRefractiveIndex()` method.
        dtype (torch.dtype): The data type for the PyTorch tensors.
        device (torch.device): The device (e.g., CPU or GPU) where the tensors are allocated.
        num_wavelength (int): The number of wavelength points used in simulations, 
            determined by the first dispersion model in the list.
    """

    def __init__(self,
                 dispersion: List[BaseDispersion],
                 num_wavelength: int,
                 dtype: torch.dtype,
                 device: torch.device,
    ) -> None:
        """
        Initialize a BaseMaterial instance.

        Args:
            dispersion (List[Dispersion]): A list of dispersion model instances.
            num_wavelenght (int): The number of wavelength points.
            dtype (torch.dtype): The desired data type for tensor operations.
            device (torch.device): The device on which the tensors will be allocated.
        """
        self.dispersion = dispersion
        self.dtype = dtype
        self.device = device
        self.num_wavelength = num_wavelength

    def refractive_index(self) -> torch.Tensor:
        """
        Compute the overall refractive index of the material.

        This method calculates the material's refractive index by summing the refractive 
        index contributions from each dispersion model in the `dispersion` list. The summation 
        is performed element-wise over a tensor that spans the specified number of wavelengths.

        Returns:
            torch.tensor: A 1D tensor of shape (num_wavelength,) representing the computed 
                          refractive index at each wavelength.
        """
        n = torch.zeros(self.num_wavelength, dtype=self.dtype, device=self.device)
        for i, disp in enumerate(self.dispersion):
            n += disp.getRefractiveIndex()

        return n
