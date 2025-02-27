from typing import List, Tuple, Literal
from .layer import BaseLayer, LayerType
from .t_matrix import T_matrix
from .optical_calculator import OpticalCalculator
import torch 

class Model:
    """
    A Model for computing the optical response of a multilayer structure using the T-matrix formalism.

    This class encapsulates an optical model composed of three parts:
      - An environment layer (env), representing the incident medium.
      - A substrate layer (subs), representing the bottom medium.
      - A list of intermediate layers (structure) that form the optical stack.
    
    The model computes the overall transfer matrices for s- and p-polarizations at given wavelengths
    and angles, then packages the results into an OpticalProperties object.

    Attributes:
        dtype (torch.dtype): Data type for tensor computations.
        device (torch.device): Device (e.g., CPU or GPU) for tensor computations.
        T_matrix (T_matrix): An instance of the T_matrix class for computing interface and layer matrices.
        env (BaseLayer): The environment layer (incident medium) with type 'env'.
        structure (List[BaseLayer]): A list of layers (typically coherent layers) forming the optical stack.
        subs (BaseLayer): The substrate layer (transmission medium) with type 'subs'.
    """

    def __init__(
            self,
            env: BaseLayer,
            structure: List[BaseLayer],
            subs: BaseLayer,
            dtype: torch.dtype,
            device: torch.device
    ) -> None:
        """
        Initialize the optical model with the environment, structure, and substrate layers.

        Args:
            env (BaseLayer): The environment layer. Its type must be 'env'.
            structure (List[BaseLayer]): A list of layers that form the optical stack.
                These layers should not be of type 'env' or 'subs'.
            subs (BaseLayer): The substrate layer. Its type must be 'subs'.
            dtype (torch.dtype): Data type for tensor operations.
            device (torch.device): Device on which tensor operations are performed.
        
        Raises:
            AssertionError: If the env or subs layers do not have the correct types,
                or if any layer in the structure has an invalid type.
        """
        self.dtype = dtype
        if self.dtype == torch.complex64:
            self.dtype_real = torch.float32
        else:
            self.dtype_real = torch.float64

        self.device = device
        self.T_matrix = T_matrix(self.dtype, self.device)

        # Load the complete model components.
        self.structure = structure
        self.env = env
        self.subs = subs

        # The environment layer must be labeled 'env' and the substrate layer 'subs'.
        assert self.env.type == 'env', 'The environment layer type is incorrect'
        assert self.subs.type == 'subs', 'The substrate layer type is incorrect'
        # Ensure no layer in the structure is an environment or substrate.
        for i, layer in enumerate(self.structure):
            assert layer.type != 'env', 'No environment layer should be in the structure'
            assert layer.type != 'subs', 'The substrate layer should not be in the structure'

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
        return (f"Model(\n"
                f"  Environment: {env_repr},\n"
                f"  Structure: {structure_repr} (n={len(self.structure)} layers),\n"
                f"  Substrate: {subs_repr},\n"
                f"  Dtype: {self.dtype}, Device: {self.device}\n"
                f")")

    def evaluate(self, wavelengths: torch.Tensor, angles: torch.Tensor) -> OpticalCalculator:
        """
        Evaluate the optical properties of the model at given wavelengths and angles.

        This method computes the refractive indices for the environment and substrate,
        and then calculates the transfer matrices for s- and p-polarizations across the complete structure.
        The resulting transfer matrices, along with the refractive indices, are packaged into an OpticalProperties object.

        Args:
            wavelengths (torch.Tensor): A tensor of wavelengths at which to evaluate the model.
            angles (torch.Tensor): A tensor of angles (in degree) of incidence.

        Returns:
            OpticalProperties: An object containing the s- and p-polarization transfer matrices,
                               and the refractive indices of the environment and substrate.
        """
        # unpack quantities and transfer to the correct device and data type
        angles = torch.deg2rad(angles.to(self.dtype_real)).to(self.dtype).to(self.device)
        wavelengths = wavelengths.to(self.dtype).to(self.device)
        angles = angles.to(self.dtype).to(self.device)
        n_env = self.env.material.refractive_index(wavelengths).to(self.dtype).to(self.device)
        n_subs = self.subs.material.refractive_index(wavelengths).to(self.dtype).to(self.device)
        n_air = torch.ones_like(n_env, dtype=self.dtype, device=self.device)
        
        # check for correct input shape 
        assert wavelengths.ndim == 1, 'Wavelengths must be a 1D tensor'
        assert angles.ndim == 1, 'Angles must be a 1D tensor'
        assert n_env.ndim == 1, 'Refractive index of the environment must be a 1D tensor'
        assert n_subs.ndim == 1, 'Refractive index of the substrate must be a 1D tensor'
        assert wavelengths.shape[0] == n_env.shape[0], 'Wavelengths and refractive index of the environment must have the same length'
        assert wavelengths.shape[0] == n_subs.shape[0], 'Wavelengths and refractive index of the substrate must have the same length'
        
        nx = n_env[:, None] * torch.sin(angles[None, :])

        # s-polarization
        T_env_s = self.T_matrix.interface_s(n_env, n_air, nx)
        T_structure_s = self.structure_matrix(wavelengths, angles, nx, pol='s')
        T_subs_s = self.T_matrix.interface_s(n_air, n_subs, nx)
        T_s = torch.einsum('...ij,...jk->...ik', T_env_s,
                             torch.einsum('...ij,...jk->...ik', T_structure_s, T_subs_s))

        # p-polarization
        T_env_p = self.T_matrix.interface_p(n_env, n_air, nx)
        T_structure_p = self.structure_matrix(wavelengths, angles, nx, pol='p')
        T_subs_p = self.T_matrix.interface_p(n_air, n_subs, nx)
        T_p = torch.einsum('...ij,...jk->...ik', T_env_p,
                             torch.einsum('...ij,...jk->...ik', T_structure_p, T_subs_p))

        return OpticalCalculator(Tm_s=T_s, Tm_p=T_p, n_env=n_env, n_subs=n_subs, nx=nx)

    def structure_matrix(self, wavelengths: torch.Tensor, angles: torch.Tensor, nx: torch.Tensor, pol: str) -> torch.Tensor:
        """
        Compute the overall transfer matrix for the layered structure.

        The transfer matrix is computed by sequentially multiplying the individual
        layer matrices. Each layer matrix is obtained by applying the coherent layer
        formula from the T_matrix class for the specified polarization.

        Args:
            wavelengths (torch.Tensor): A tensor of wavelengths at which to compute the matrix.
            angles (torch.Tensor): A tensor of angles (in radians) of incidence.
            nx (torch.Tensor): The x-component of the wave vector, computed as n_env * sin(angle).
            pol (str): Polarization, either 's' or 'p', specifying which interface formula to use.

        Returns:
            torch.Tensor: The overall transfer matrix for the structure.
        """
        # Initialize the structure matrix as an identity matrix for each wavelength and angle.
        T_structure = torch.eye(2, dtype=self.dtype, device=self.device)
        T_structure = T_structure.unsqueeze(0).unsqueeze(0).repeat(wavelengths.shape[0], angles.shape[0], 1, 1)

        # Multiply the transfer matrix for each layer in the structure.
        for i, layer in enumerate(self.structure):
            #check input data shape and transform to correct data type and device 
            d = layer.thickness.to(self.dtype_real).to(self.device)
            n = layer.material.refractive_index(wavelengths).to(self.dtype).to(self.device)
            assert n.ndim == 1, 'Refractive index must be a 1D tensor'
            assert d.ndim == 0, 'Thickness must be a scalar tensor'
            assert wavelengths.shape[0] == n.shape[0], 'Wavelengths and refractive index must have the same length'
            
            T_layer = self.T_matrix.coherent_layer(pol=pol, n=n, d=d, wavelengths=wavelengths, nx=nx)
            T_structure = torch.einsum('...ij,...jk->...ik', T_structure, T_layer)

        return T_structure
