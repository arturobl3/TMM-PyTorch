import torch
import numpy as np
from transfer_matrix import *

def test_interface_matrix():
    """
    Test the boundary matrix between air (n=1) and glass (n=1.5) at normal incidence.
    Expected reflection coefficient R = ((1 - 1.5) / (1 + 1.5))^2 = 0.04
    Expected transmission coefficient T = 1 - R = 0.96
    """
    n1 = torch.tensor(1.0 + 0j)  # Air
    n2 = torch.tensor(1.5 + 0j)  # Glass
    wavelength = 500e-9  # 500 nm
    theta1 = torch.tensor(0.0)  # Normal incidence
    theta2 = torch.tensor(0.0)

    T_matrix = interface_matrix_s_pol(n1, n2, k_x=0)
    
    r = T_matrix[1, 0] / T_matrix[0, 0]
    t = 1 / T_matrix[0, 0]

    R = torch.abs(r) ** 2
    T = torch.abs(t) ** 2 * torch.real(n2 * torch.cos(theta2)) / torch.real(n1 * torch.cos(theta1))

    assert torch.allclose(R, torch.tensor(0.04), atol=1e-3), f"Failed: R = {R}"
    assert torch.allclose(T, torch.tensor(0.96), atol=1e-3), f"Failed: T = {T}"
    print("Boundary matrix test passed!")


def test_propagation_matrix():
    """
    Test the propagation matrix for a thin glass layer (n=1.5) with d=100 nm at λ=500 nm.
    The phase shift should be 2π * n * d / λ.
    """
    n = torch.tensor(1.5 + 0j)
    d = 100e-9  # 100 nm
    wavelength = 500e-9  # 500 nm
    theta = torch.tensor(0.0)

    P_matrix = propagation_matrix(n, d, 0, wavelength)
    
    expected_phase_shift = (2 * np.pi * n * d / wavelength).real
    calculated_phase_shift = torch.angle(P_matrix[0, 0]).item()

    assert np.isclose(calculated_phase_shift, -expected_phase_shift, atol=1e-3), f"Failed: Phase shift = {calculated_phase_shift}, expected = {expected_phase_shift}"
    print("Propagation matrix test passed!")

# def test_transfer_matrix():
#     """
#     Test the transfer matrix for a single glass layer (n=1.5) with d=100 nm at λ=500 nm,
#     surrounded by air on both sides. Compare with known analytical reflectance.
#     """
#     materials = torch.tensor([[ 1.5 + 0j]])  # Air - Glass - Air
#     thicknesses = torch.tensor([ 100e-9])  # Thicknesses (air layers ignored)
#     wavelengths = torch.tensor([500e-9])  # 500 nm
#     angles = torch.tensor([0.0])  # Normal incidence

#     M_total = transfer_matrix(materials, thicknesses, wavelengths, angles)
    
#     r = M_total[0, 0, 1, 0] / M_total[0, 0, 0, 0]
#     t = 1 / M_total[0, 0, 0, 0]

#     R = torch.abs(r)**2
#     T = torch.abs(t)**2

#     assert torch.allclose(R + T, torch.tensor(1.0), atol=1e-3), f"Failed: R + T = {R + T}"
#     print("Transfer matrix test passed!")


# def test_multilayer_structure():
#     """
#     Test a multilayer structure:
#     Air (n=1) → Glass (n=1.5, 100 nm) → TiO2 (n=2.5, 200 nm) → Air (n=1)
    
#     Expected behavior:
#     - Energy conservation: R + T ≈ 1
#     - Transmission should be nonzero since there is no perfect reflection.
#     """
#     # Define the materials and thicknesses
#     materials = torch.tensor([[1.5 + 0j, 2.5 + 0j]])  # Air - Glass - TiO2 - Air
#     thicknesses = torch.tensor([100e-9, 200e-9])  # Air ignored
#     wavelengths = torch.tensor([500e-9])  # 500 nm
#     angles = torch.tensor([0.0])  # Normal incidence

#     # Compute the total transfer matrix
#     M_total = transfer_matrix(materials, thicknesses, wavelengths, angles)

#     # Compute reflection and transmission
#     r = M_total[0, 0, 1, 0] / M_total[0, 0, 0, 0]
#     t = 1 / M_total[0, 0, 0, 0]

#     R = torch.abs(r) ** 2
#     T = torch.abs(t) ** 2 

#     # Check that R + T ≈ 1 (energy conservation)
#     assert torch.allclose(R + T, torch.tensor(1.0), atol=1e-3), f"Failed: R + T = {R + T}"
    
#     print("Multilayer structure test passed!")


def test_1_layer_analytical(n1,n2,n3, wavelengths, theta_1, d):

    wavelengths = wavelengths.unsqueeze(1)
    theta_1 = theta_1.unsqueeze(0)
    theta_1 = torch.deg2rad(theta_1) 
    theta_2 = snell_law_vectorized(n1, n2, theta_1)
    theta_3 = snell_law_vectorized(n2, n3, theta_2)

    k0 = 2*torch.pi/wavelengths
    k1z = n1 * k0 * torch.cos(theta_1)
    k2z = n2 * k0 *  torch.cos(theta_2)
    k3z = n3 * k0 *  torch.cos(theta_3)

    r12 = (k1z - k2z) / (k1z + k2z)
    r23 = (k2z - k3z) / (k2z + k3z)
    beta = k2z*d

    t12 = (2 * k1z) / (k1z + k2z)
    t23 = (2 * k2z) / (k2z + k3z)

    r = (r12 + r23 * torch.exp(2j*beta)) / (1 + r12 * r23 * torch.exp(2j*beta))
    t = (t12 * t23 * torch.exp(1j*beta)) / (1 + r12 * r23 * torch.exp(2j*beta))

    R = torch.abs(r) ** 2
    T = (torch.abs(t) ** 2 *
            torch.real(n3 * torch.cos(theta_3)) /
            torch.real(n1 * torch.cos(theta_1)))
    
    return R,T


def one_layer_rt(n1, n2, n3, wavelengths, theta_1_deg, d):
    """
    Computes the complex reflection (r) and transmission (t)
    coefficients for a single layer (with refractive index n2 and thickness d)
    sandwiched between media with indices n1 and n3.
    Parameters:
      n1, n2, n3 : torch.Tensor or float
         Refractive indices for incident, layer, and exit media.
      wavelengths : torch.Tensor
         Wavelengths (in meters); assumed shape (num_wavelengths,).
      theta_1_deg : torch.Tensor or float
         Incident angle in medium 1 (in degrees); assumed shape (num_angles,) or scalar.
      d : float or torch.Tensor
         Thickness of the layer.
    Returns:
      r, t : tuple of torch.Tensor
         Complex reflection and transmission coefficients.
    """
    # Ensure wavelengths and angles have extra dimensions for broadcasting:
    wavelengths = wavelengths.unsqueeze(1)  # shape (num_wavelengths, 1)
    theta_1 = torch.deg2rad(theta_1_deg).unsqueeze(0)  # shape (1, num_angles)
    
    # Compute propagation angles in the layer and exit medium
    theta_2 = snell_law_vectorized(n1, n2, theta_1)
    theta_3 = snell_law_vectorized(n2, n3, theta_2)
    
    # Wavevector in vacuum:
    k0 = 2 * torch.pi / wavelengths  # shape (num_wavelengths, 1)
    
    # z-components in each medium:
    k1z = n1 * k0 * torch.cos(theta_1)
    k2z = n2 * k0 * torch.cos(theta_2)
    k3z = n3 * k0 * torch.cos(theta_3)
    
    # Fresnel coefficients at the first and second interfaces (s-polarization)
    r12 = (k1z - k2z) / (k1z + k2z)
    r23 = (k2z - k3z) / (k2z + k3z)
    
    t12 = (2 * k1z) / (k1z + k2z)
    t23 = (2 * k2z) / (k2z + k3z)
    
    # Phase accumulation in the layer
    beta = k2z * d  # shape (num_wavelengths, 1)
    
    # Combine to get overall r and t for the single layer:
    r = (r12 + r23 * torch.exp(2j * beta)) / (1 + r12 * r23 * torch.exp(2j * beta))
    t = (t12 * t23 * torch.exp(1j * beta)) / (1 + r12 * r23 * torch.exp(2j * beta))
    
    return r, t

def recursive_multilayer(n_list, wavelengths, theta_1, d_list):
    """
    Recursively computes the effective reflection and transmission
    coefficients for a multilayer stack.
    
    Parameters:
      n_list : list of torch.Tensor or float
         Refractive indices at each interface. For a structure with m layers,
         n_list should have length m+2, e.g. [n_incident, n_layer1, n_layer2, ..., n_exit].
      wavelengths : torch.Tensor
         Wavelengths (in meters), shape (num_wavelengths,).
      theta_1_deg : torch.Tensor or float
         Incident angle in the first medium (in degrees), shape (num_angles,) or scalar.
      d_list : list of float or torch.Tensor
         List of layer thicknesses. Should have length equal to the number of layers.
    
    Returns:
      r, t : tuple of torch.Tensor
         Complex effective reflection and transmission coefficients of the entire structure.
    """
    # Base case: one layer (three media)
    if len(n_list) == 3:
        return one_layer_rt(n_list[0], n_list[1], n_list[2], wavelengths, theta_1, d_list[0])
    
    # Recursive case: more than one layer.
    # The idea is to compute the effective coefficients of the stack
    # from the second layer onward, then combine with the first layer.
    
    # Compute the reflection/transmission of the rest of the stack:
    # For the first interface, the incident angle is theta_1_deg.
    # Compute the angle inside the first layer:
    theta_2 = snell_law_vectorized(n_list[0], n_list[1], theta_1)
    # (Make sure to convert theta_2 back to degrees if needed for the recursive call.)
    
    # Recursively compute effective r and t for the stack starting at n_list[1]
    # and with the remaining thicknesses d_list[1:].
    r_eff, t_eff = recursive_multilayer(n_list[1:], wavelengths, theta_2, d_list[1:])
    
    # Now, combine the first layer with the effective stack.
    # Use the single-layer formula where n1 = n_list[0], n2 = n_list[1] and
    # n3 is now the effective exit medium. Here, we “simulate” an interface between
    # the first layer and the effective medium.
    #
    # For this, we treat r_eff and t_eff as the effective reflection/transmission
    # coefficients for an interface between n_list[1] and the effective medium.
    #
    # We need to build a “composite” layer using the first thickness d_list[0].
    wavelengths_expanded = wavelengths.unsqueeze(1)  # shape (num_wavelengths, 1)
    theta_1_expanded = theta_1.unsqueeze(0)  # shape (1, num_angles)
    k0 = 2 * torch.pi / wavelengths_expanded
    
    k1z = n_list[0] * k0 * torch.cos(theta_1_expanded)
    k2z = n_list[1] * k0 * torch.cos(theta_2.unsqueeze(0))
    
    r12 = (k1z - k2z) / (k1z + k2z)
    t12 = (2 * k1z) / (k1z + k2z)
    
    # Compute phase shift in the first layer:
    beta = k2z * d_list[0]
    
    # Now, interpret r_eff as the effective reflection coefficient at the second interface.
    # Combine using the thin-film formulas:
    r = (r12 + r_eff * torch.exp(2j * beta)) / (1 + r12 * r_eff * torch.exp(2j * beta))
    t = t12 * t_eff * torch.exp(1j * beta) / (1 + r12 * r_eff * torch.exp(2j * beta))
    
    return r, t

# # Example usage:
# if __name__ == "__main__":
#     # Define a simple multilayer stack:
#     # For example, air (n=1.0), layer 1 (n=1.5), layer 2 (n=2.0), substrate (n=1.45)
#     n_list = [1.0, 1.5, 2.0, 1.45]
#     # Thicknesses for each layer (here two layers)
#     d_list = [100e-9, 200e-9]  # in meters
#     # Wavelengths (say 400 nm to 700 nm)
#     wavelengths = torch.linspace(400e-9, 700e-9, 300)
#     # Incident angles (for example, 0° to 30°)
#     theta_inc_deg = torch.tensor([0.0, 10.0, 20.0, 30.0])
    
#     # Compute effective reflection and transmission (complex amplitudes)
#     r_total, t_total = recursive_multilayer(n_list, wavelengths, theta_inc_deg, d_list)
    
#     # Compute intensities (reflectance and transmittance)
#     R_total = torch.abs(r_total)**2
#     T_total = torch.abs(t_total)**2
    
#     print("Reflectance shape:", R_total.shape)
#     print("Transmittance shape:", T_total.shape)
