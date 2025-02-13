import torch
import numpy as np
# Constants
c = 3e8  # Speed of light in vacuum (m/s)

def snell_law(n1, n2, angle_incidence):
    """
    Applies Snell's law to calculate the angle of propagation in a layer.
    n1: Refractive index of the current layer
    n2: Refractive index of the next layer
    angle_incidence: Angle of incidence in the current layer (radians)
    """
    return torch.arcsin((n1 / n2) * torch.sin(angle_incidence))

def propagation_matrix(n_i, d_i, wavelength, theta_i):
    """
    Computes the propagation transfer matrix through a layer.
    n_i: Refractive index of the layer
    d_i: Thickness of the layer
    wavelength: Wavelength of light
    theta_i: Propagation angle in the layer
    """
    delta_i = (2 * np.pi * n_i * d_i / wavelength) * torch.cos(theta_i)
    return torch.tensor([[torch.exp(-1j * delta_i), 0],
                         [0, torch.exp(1j * delta_i)]], dtype=torch.complex64)

def interface_matrix(n_i, n_next, theta_i, theta_next, wavelength):
    """
    Computes the boundary transfer matrix between two media.
    n_i: Refractive index of the current layer
    n_next: Refractive index of the next layer
    theta_i: Angle of propagation in the current layer
    theta_next: Angle of propagation in the next layer
    """
    k_0 = 2*np.pi/wavelength
    k_1z = n_i *k_0* torch.cos(theta_i)
    k_2z = n_next *k_0* torch.cos(theta_next)

    r_ij = ((k_1z - k_2z) /
            (k_1z + k_2z))
    t_ij = (2 * k_1z /
            (k_1z + k_2z))
    
    return torch.tensor([[1 / t_ij, r_ij / t_ij],
                         [r_ij / t_ij, 1 / t_ij]], dtype=torch.complex64)


def transfer_matrix(n, d, wavelengths, angles_of_incidence):
    """
    Compute the transfer matrix for the entire stack at each wavelength and angle of incidence.
    Processes all layers, including the first and last, without assuming they are air.
    """
    num_wavelengths = len(wavelengths)
    num_angles = len(angles_of_incidence)
    M_total = torch.zeros((num_wavelengths, num_angles, 2, 2), dtype=torch.complex64)

    for w_idx, wlg in enumerate(wavelengths):
        for a_idx, alpha in enumerate(angles_of_incidence):
            # angle_rad = torch.tensor(np.deg2rad(alpha), dtype=torch.float32)
            angle_rad = torch.deg2rad(alpha)
            M_current = torch.eye(2, 2, dtype=torch.complex64)

            for i in range(len(n) - 1):
                n_i, n_next = n[i, w_idx], n[i + 1, w_idx]
                d_i = d[i] if i < len(d) else 0  # Ensure thickness array does not exceed bounds
                
                # Compute the propagation angle using Snell's law
                theta_i = snell_law(n[i - 1, w_idx] if i > 0 else n_i, n_i, angle_rad if i == 0 else theta_prev)
                theta_next = snell_law(n_i, n_next, theta_i)

                # Compute boundary matrices
                T_in = interface_matrix(n[i - 1, w_idx] if i > 0 else n_i, n_i, angle_rad if i == 0 else theta_prev, theta_i, wlg)
                P_i = propagation_matrix(n_i, d_i, wlg, theta_i)
                T_out = interface_matrix(n_i, n_next, theta_i, theta_next, wlg)
                
                # Compute total matrix for the layer: T_in * P_i * T_out
                M_layer = torch.matmul(T_in, torch.matmul(P_i, T_out))
                M_current = torch.matmul(M_current, M_layer)
                
                theta_prev = theta_next  # Store angle for next iteration

            M_total[w_idx, a_idx] = M_current

    return M_total

def reflectance_transmittance(M_total, n):
    """
    Calculate the reflectance and transmittance using the transfer matrices.
    M_total: Transfer matrix of shape (num_wavelengths, num_angles, 2, 2)
    n: Refractive indices of the layers
    wavelengths: Wavelengths of light
    """
    num_wavelengths, num_angles = M_total.shape[:2]
    R = torch.zeros(num_wavelengths, num_angles)
    T = torch.zeros(num_wavelengths, num_angles)

    # Calculate reflectance and transmittance
    for w_idx in range(num_wavelengths):
        for a_idx in range(num_angles):
            M = M_total[w_idx, a_idx]

            # Reflection and transmission coefficients
            r = M[1, 0] / M[0, 0]
            t = 1 / M[0, 0]
            # t = torch.linalg.det(M) / M[0,0]

            R[w_idx, a_idx] = torch.abs(r)**2
            T[w_idx, a_idx] = torch.abs(t)**2 * torch.real(n[-1, w_idx] / n[0, w_idx])

    return R, T

