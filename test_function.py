import torch
import numpy as np
from transfer_matrix import *

def test_interface_matrix():
    """
    Test the boundary matrix between air (n=1) and glass (n=1.5) at normal incidence.
    Expected reflection coefficient R = ((1 - 1.5) / (1 + 1.5))^2 = 0.04
    Expected transmission coefficient T = 1 - R = 0.96
    """
    n1 = 1.0 + 0j  # Air
    n2 = 1.5 + 0j  # Glass
    wavelength = 500e-9  # 500 nm
    theta1 = torch.tensor(0.0)  # Normal incidence
    theta2 = torch.tensor(0.0)

    T_matrix = interface_matrix_s_pol(n1, n2, theta1, theta2)
    
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
    n = 1.5 + 0j
    d = 100e-9  # 100 nm
    wavelength = 500e-9  # 500 nm
    theta = torch.tensor(0.0)

    P_matrix = propagation_matrix(n, d, wavelength, theta)
    
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
    t = (t12 * t23 * torch.exp(2j*beta)) / (1 + r12 * r23 * torch.exp(2j*beta))

    R = torch.abs(r) ** 2
    T = (torch.abs(t) ** 2 *
            torch.real(n3 * torch.cos(theta_3)) /
            torch.real(n1 * torch.cos(theta_1)))
    
    return R,T

if __name__ == "__main__":
    test_interface_matrix()
    test_propagation_matrix()
    print("All tests passed successfully!")
