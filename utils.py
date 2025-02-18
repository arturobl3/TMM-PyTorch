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

# def forward_kz(n, kx):
#     """
#     Ensure the correct computation of kz.
#     The function check the sign of the real part of kz and ensure that the wave is propagating forward.
#     """
#     # principal sqrt
#     kz = torch.sqrt(n**2 - kx**2 + 0j)
#     # force real(kz) >= 0
#     mask = torch.real(kz) < 0
#     kz[mask] = -kz[mask]
#     return kz
def forward_kz(n, kx):
    """
    Picks the branch of sqrt(n^2 - kx^2) so that:
      - traveling wave:  Re(kz) >= 0
      - evanescent wave: Im(kz) > 0  (decays in +z)
    """
    kz = torch.sqrt(n**2 - kx**2 + 0j)

    # For traveling waves: if Re(kz) < 0, flip sign
    mask_travel = (torch.real(kz) < 0) & (torch.imag(kz) == 0)
    kz[mask_travel] = -kz[mask_travel]

    # For evanescent waves: if Im(kz) < 0, flip sign
    mask_evan = (torch.imag(kz) < 0) & (torch.real(kz) == 0)
    kz[mask_evan] = -kz[mask_evan]

    return kz


def snell_law_vectorized(n1, n2, theta1):
    """
    Vectorized Snell's law:
    n1, n2, theta1 each have shape (num_wavelengths, num_angles) or broadcastable.
    Returns theta2 with same shape, applying arcsin(n1/n2 * sin(theta1)) in parallel.
    """
    sin_theta2 = (torch.real(n1) / torch.real(n2)) * torch.sin(theta1)
    sin_theta2_clamped = torch.clamp(sin_theta2, -1.0, 1.0)
    theta2 = torch.arcsin(sin_theta2_clamped)
    return theta2