import torch
from typing import List
from .dispersion import BaseDispersion

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
        name (str): the name of the implemented material
        dtype (torch.dtype): The data type for the PyTorch tensors.
        device (torch.device): The device (e.g., CPU or GPU) where the tensors are allocated.
    """

    def __init__(self,
                 dispersion: List[BaseDispersion],
                 name : str = None,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Initialize a BaseMaterial instance.

        Args:
            dispersion (List[Dispersion]): A list of dispersion model instances.
            name (str): The name of the material implemented
            dtype (torch.dtype): The desired data type for tensor operations.
            device (torch.device): The device on which the tensors will be allocated.
        """
        self.dispersion = dispersion
        self.dtype = dtype
        self.device = device
        self.name = name

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the overall refractive index of the material.

        This method calculates the material's refractive index.

        Parameters:
            wavelengths: torch.Tensor.

        Returns:
            torch.tensor: A 1D tensor of shape (num_wavelength,) representing the computed 
                          refractive index at each wavelength.
        """
        epsilon = self.epsilon(wavelengths)
        n = torch.sqrt(epsilon)
        return n
    
    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Compute the overall epsilon of the material.

        This method calculates the material's epsilon by summing the dielectric permittivity
        contributions from each dispersion model in the `dispersion` list. The summation 
        is performed element-wise over a tensor that spans the specified number of wavelengths.

        Parameters:
            wavelengths: torch.Tensor. 

        Returns:
            torch.tensor: A 1D tensor of shape (num_wavelength,) representing the computed 
                          refractive index at each wavelength.
        """
        epsilons_list =[disp.epsilon(wavelengths) for disp in self.dispersion]
        epsilon = torch.stack(epsilons_list, dim=0).sum(dim=0)
        return epsilon
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Material instance.

        Returns:
            str: A string summarizing the Layer.
        """

        dispersion_repr = f"[{', '.join(repr(dispersion) for dispersion in self.dispersion)}]"
        return (f"Material(\n"
                f"  Name: {self.name},\n"
                f"  Dispersions: {dispersion_repr} (n={len(self.dispersion)} dispersions),\n"
                f"  Dtype: {self.dtype}, Device: {self.device}\n"
                f")")
