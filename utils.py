import torch

def refractive_index_function(materials, wavelengths):
    """
    Defines the refractive indices for different layers as a function of wavelength.
    Replace the refractive indices with experimental dispersion data for materials.
    """
    refractive_indices = torch.zeros((len(materials), len(wavelengths)), dtype=torch.complex64)

    for i, material in enumerate(materials):
        if material == "air":
            refractive_indices[i, :] = 1.0 + 0j  # Air
        elif material == "SiO2":
            refractive_indices[i, :] = 1.45 + 0j  
        elif material == "TiO2":
            refractive_indices[i, :] = 2.5 + 0j  

    return refractive_indices