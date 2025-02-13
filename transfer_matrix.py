import torch
import numpy as np
# Constants
c = 3e8  # Speed of light in vacuum (m/s)

def propagation_matrix(n_i, d_i, wavelength, theta_i):
    """
    Computes the propagation transfer matrix for s-polarization through a layer,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of the layer. Shape: (num_wavelengths, 1)
    d_i : torch.Tensor
        Thickness of the layer. Must be broadcastable to n_i. E.g. shape: (num_wavelengths, 1) or (1,) or matching n_i
    wavelength : torch.Tensor
        Wavelength of light. Shape: (num_wavelengths, 1) or broadcastable with n_i
    theta_i : torch.Tensor
        Propagation angle in the layer. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    torch.Tensor
        Propagation matrices of shape (num_wavelengths, num_angles, 2, 2)
    """
    # Phase shift delta_i for each wavelength & angle
    delta_i = (2 * np.pi * n_i * d_i / wavelength) * torch.cos(theta_i)

    # Create an empty tensor for the result
    M_shape = delta_i.shape + (2, 2)
    M = torch.zeros(M_shape, dtype=torch.complex64)

    # Fill in the diagonal terms
    M[..., 0, 0] = torch.exp(-1j * delta_i)
    M[..., 1, 1] = torch.exp(1j * delta_i)

    return M


def interface_matrix_s_pol(n_i, n_next, theta_i, theta_next):
    """
    Computes the boundary (interface) transfer matrix between two media for s-polarization,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of current layer. Shape: (num_wavelengths, num_angles)
    n_next : torch.Tensor
        Refractive index of next layer. Same shape as n_i
    theta_i : torch.Tensor
        Angle in current layer. Shape: (num_wavelengths, num_angles)
    theta_next : torch.Tensor
        Angle in the next layer. Same shape as theta_i
    Returns
    -------
    torch.Tensor
        Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
    """

    # Wavevector components in each layer
    k_1z = n_i * torch.cos(theta_i)
    k_2z = n_next * torch.cos(theta_next)

    r_ij = (k_1z - k_2z) / (k_1z + k_2z)
    t_ij = (2 * k_1z) / (k_1z + k_2z)

    # Construct the tensor for the result
    M_shape = r_ij.shape + (2, 2)  # shape is (num_wavelengths, num_angles, 2, 2)
    M = torch.zeros(M_shape, dtype=torch.complex64)

    # Fill the matrix elements:

    M[..., 0, 0] = 1 / t_ij
    M[..., 0, 1] = r_ij / t_ij
    M[..., 1, 0] = r_ij / t_ij
    M[..., 1, 1] = 1 / t_ij

    return M

def interface_matrix_p_pol(n_i, n_next, theta_i, theta_next):
    """
    Computes the boundary (interface) transfer matrix between two media for p-polarization,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of current layer. Shape: (num_wavelengths, num_angles)
    n_next : torch.Tensor
        Refractive index of next layer. Same shape as n_i
    theta_i : torch.Tensor
        Angle in current layer. Shape: (num_wavelengths, num_angles)
    theta_next : torch.Tensor
        Angle in the next layer. Same shape as theta_i
    Returns
    -------
    torch.Tensor
        Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
    """

    # Wavevector components in each layer
    k_1z = n_i * torch.cos(theta_i)
    k_1p = n_next * torch.cos(theta_i)
    k_2p = n_i * torch.cos(theta_next)

    r_ij = (k_1p - k_2p) / (k_1p + k_2p)
    t_ij = (2 * k_1z) / (k_1p + k_2p)

    # Construct the tensor for the result
    M_shape = r_ij.shape + (2, 2)  # shape is (num_wavelengths, num_angles, 2, 2)
    M = torch.zeros(M_shape, dtype=torch.complex64)

    # Fill the matrix elements:

    M[..., 0, 0] = 1 / t_ij
    M[..., 0, 1] = r_ij / t_ij
    M[..., 1, 0] = r_ij / t_ij
    M[..., 1, 1] = 1 / t_ij

    return M


def reflectance_transmittance(M, n_incident, n_exit, theta_incident, theta_exit):
    """
    Extract reflection and transmission coefficients from the transfer matrix M
    in parallel for all wavelengths and angles.

    Parameters
    ----------
    M : torch.Tensor
        Transfer matrix of shape (num_wavelengths, num_angles, 2, 2)
    n_incident : torch.Tensor
        Refractive index of the first (incident) medium, shape (num_wavelengths, num_angles)
        or broadcastable to that shape
    n_exit : torch.Tensor
        Refractive index of the last (exit) medium, shape (num_wavelengths, num_angles)
        or broadcastable to that shape
    theta_incident : torch.Tensor
        Incident angles in the first medium, shape (num_wavelengths, num_angles)
    theta_exit : torch.Tensor
        Transmitted angles in the exit medium, shape (num_wavelengths, num_angles)

    Returns
    -------
    R : torch.Tensor
        Reflectance for each wavelength and angle. Shape: (num_wavelengths, num_angles)
    T : torch.Tensor
        Transmittance for each wavelength and angle. Same shape: (num_wavelengths, num_angles)
    """
    # Reflection and transmission coefficients from the matrix
    r = M[..., 1, 0] / M[..., 0, 0]
    t = 1.0 / M[..., 0, 0]

    # Reflectance
    R = torch.abs(r) ** 2

    # Transmittance (for s-polarization), accounting for angle dependence
    # T = |t|^2 * ( n_exit cos(theta_exit) ) / ( n_incident cos(theta_incident) )
    T = (torch.abs(t) ** 2 * (
            torch.real(n_exit * torch.cos(theta_exit)) /
            torch.real(n_incident * torch.cos(theta_incident))))

    return R, T


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

def compute_transfer_matrix(n, d, wavelengths, angles, interface_fn, propagation_fn):
    """
    Computes the total transfer matrix for a multilayer structure 
    over all wavelengths and angles in parallel (s-polarization).

    Parameters
    ----------
    n : torch.Tensor
        Refractive indices of shape (num_layers, num_wavelengths), 
        or broadcastable to that shape. E.g. each layer has n[i, :] over wavelengths.
    d : torch.Tensor
        Thicknesses of shape (num_layers, ), or broadcastable to (num_layers, num_wavelengths).
    wavelengths : torch.Tensor
        Wavelengths of shape (num_wavelengths, ), or (num_wavelengths, 1).
    angles : torch.Tensor
        Incidence angles (in radians) or (in degrees, if you convert inside).
        Shape: (num_angles,)
    interface_fn : function
        Vectorized interface function, e.g. interface_matrix_s_pol
    propagation_fn : function
        Vectorized propagation function
    snell_fn : function
        Vectorized snell law function
    Returns
    -------
    M_total : torch.Tensor
        Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
    """

    # Prepare shapes
    num_layers = n.shape[0]           
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[0]

    # Convert angles to radians if needed and broadcast
    angles_rad = torch.deg2rad(angles) if angles.max() > np.pi else angles
    # Expand dims to shape (num_wavelengths, num_angles) for vector ops if needed
    angles_rad = angles_rad.unsqueeze(0).expand(num_wavelengths, num_angles)

    # Track the 'current angle' in each layer 
    theta_current = angles_rad

    # Initialize total transfer matrix as identity: shape (num_wavelengths, num_angles, 2, 2)
    M_current = torch.eye(2, 2, dtype=torch.complex64).repeat(num_wavelengths, num_angles, 1, 1)

    for i in range(num_layers - 1):
        # Indices in the n and d arrays
        n_i     = n[i, :]       
        n_next  = n[i + 1, :]   
        d_i     = d[i] if i < len(d) else 0.0

        # Expand n_i, n_next, and d_i so they can broadcast with angles
        n_i2d    = n_i.unsqueeze(1).expand(num_wavelengths, num_angles)
        n_next2d = n_next.unsqueeze(1).expand(num_wavelengths, num_angles)

        # Current thickness broadcast
        d_i_val = d_i 
        d_i_val = d_i_val.unsqueeze(1).expand(num_wavelengths, num_angles)

        # 1) Snell's law for layer i -> i+1
        theta_next = snell_law_vectorized(n_i2d, n_next2d, theta_current)

        # 2) Build interface matrix: from layer i to layer i+1
        I_mat = interface_fn(n_i2d, n_next2d, theta_current, theta_next)

        # 3) Build propagation matrix for layer i
        P_mat = propagation_fn(n_i2d, d_i_val, 
                               wavelengths.unsqueeze(1).expand(num_wavelengths, num_angles),
                               theta_current)

        # 4) Multiply into M_current. 
        M_layer = torch.einsum('...ij,...jk->...ik', P_mat, I_mat )

        # 5) Update M_current
        M_current = torch.einsum('...ij,...jk->...ik', M_current, M_layer)

        # Move on to next layer angle
        theta_current = theta_next

    return M_current, theta_current
