import torch
import numpy as np
# Constants
c = 2,99792458e8  # Speed of light in vacuum (m/s)

def propagation_matrix(n_i, d_i, k_x, wavelength):
    """
    Computes the propagation transfer matrix for through a layer,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of the layer. Shape: (num_wavelengths, 1)
    d_i : torch.Tensor
        Thickness of the layer. Must be broadcastable to n_i. E.g. shape: (num_wavelengths, 1) or (1,) or matching n_i
    wavelength : torch.Tensor
        Wavelength of light. Shape: (num_wavelengths, 1) or broadcastable with n_i
    k_x : torch.Tensor
        Transversal component of the K-vector. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    torch.Tensor
        Propagation matrices of shape (num_wavelengths, num_angles, 2, 2)
    """
    k_2z = torch.sqrt(n_i**2 - k_x**2+0j)

    # Phase shift delta_i for each wavelength & angle
    delta_i = (2 * np.pi / wavelength) * k_2z * d_i

    # Create an empty tensor for the result
    M_shape = delta_i.shape + (2, 2)
    M = torch.zeros(M_shape, dtype=torch.complex64)

    # Fill in the diagonal terms
    M[..., 0, 0] = torch.exp(-1j * delta_i)
    M[..., 1, 1] = torch.exp(1j * delta_i)

    return M


def interface_matrix_s_pol(n_i, n_next, k_x):
    """
    Computes the boundary (interface) transfer matrix between two media for s-polarization,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of current layer. Shape: (num_wavelengths, num_angles)
    k_x : torch.Tensor
        Transversal component of the K-vector. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    torch.Tensor
        Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
    """

    # Wavevector components in each layer
    k_1z = torch.sqrt(n_i**2 - k_x**2 +0j)
    k_2z = torch.sqrt(n_next**2 - k_x**2 +0j)

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

def interface_matrix_p_pol(n_i, n_next, k_x):
    """
    Computes the boundary (interface) transfer matrix between two media for p-polarization,
    in parallel for all wavelengths and angles.
    Parameters
    ----------
    n_i : torch.Tensor
        Refractive index of current layer. Shape: (num_wavelengths, num_angles)
    n_next : torch.Tensor
        Refractive index of next layer. Same shape as n_i
    k_x : torch.Tensor
        Transversal component of the K-vector. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    torch.Tensor
        Interface matrices of shape (num_wavelengths, num_angles, 2, 2)
    """
    # Wavevector components in each layer
    k_1z = torch.sqrt(n_i**2 - k_x**2 +0j)
    k_2z = torch.sqrt(n_next**2 - k_x**2 +0j)
    
    r_ij = (n_next / n_i * k_1z - n_i / n_next * k_2z) / (n_next / n_i * k_1z + n_i / n_next * k_2z)
    t_ij = (2 * k_1z) / (n_next / n_i * k_1z + n_i / n_next * k_2z)

    # Construct the tensor for the result
    M_shape = r_ij.shape + (2, 2)  # shape is (num_wavelengths, num_angles, 2, 2)
    M = torch.zeros(M_shape, dtype=torch.complex64)

    # Fill the matrix elements:
    M[..., 0, 0] = 1 / t_ij
    M[..., 0, 1] = r_ij / t_ij
    M[..., 1, 0] = r_ij / t_ij
    M[..., 1, 1] = 1 / t_ij

    return M


def reflectance_transmittance(M, n_incident, n_exit, theta_incident):
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
    t = torch.linalg.det(M) / M[..., 0, 0]

    # Reflectance
    R = torch.abs(r) ** 2
    k_x = n_incident * torch.sin(theta_incident)
    k_1z = n_incident * torch.cos(theta_incident)
    k_2z = torch.sqrt(n_exit**2 - k_x**2 +0j)

    # Transmittance (for s-polarization), accounting for angle dependence
    # T = |t|^2 * ( k_2z/k_1z)
    T = torch.abs(t) ** 2 * torch.real(k_2z / k_1z)

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

def single_layer_transfer_matrix_s_pol(n, d, wavelengths, angles, k_x):
    """
    Computes the total transfer matrix for a single layer structure surrounded by air 
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
        Incidence angles (in degrees).
        Shape: (num_angles,) 
    k_x : torch.Tensor
        Transversal component of the K-vector. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    M_total : torch.Tensor
        Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
    """
    # Prepare shapes      
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[1]
    n_air = torch.ones((num_wavelengths, num_angles))


    # 1) Build interface matrix: from air to layer
    I_in = interface_matrix_s_pol(n_air, n, k_x)

    # 2) Build propagation matrix for layer i
    P_mat = propagation_matrix(n, d, k_x, wavelengths)

    # 3) Build interface matrix: from air to layer
    I_out = interface_matrix_s_pol(n, n_air, k_x)

    # 4) Multiply into M_current. 
    M_layer = torch.einsum('...ij,...jk->...ik', I_in, torch.einsum('...ij,...jk->...ik', P_mat, I_out )) 

    return M_layer

def single_layer_transfer_matrix_p_pol(n, d, wavelengths, angles, k_x):
    """
    Computes the total transfer matrix for a single layer structure surrounded by air 
    over all wavelengths and angles in parallel (p-polarization).
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
        Incidence angles (in degrees).
        Shape: (num_angles,) 
    k_x : torch.Tensor
        Transversal component of the K-vector. Shape: (num_wavelengths, num_angles)
    Returns
    -------
    M_total : torch.Tensor
        Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
    """
    # Prepare shapes      
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[1]
    n_air = torch.ones((num_wavelengths, num_angles))


    # 1) Build interface matrix: from air to layer
    I_in = interface_matrix_p_pol(n_air, n, k_x)

    # 2) Build propagation matrix for layer i
    P_mat = propagation_matrix(n, d, k_x, wavelengths)

    # 3) Build interface matrix: from air to layer
    I_out = interface_matrix_p_pol(n, n_air, k_x)

    # 4) Multiply into M_current. 
    M_layer = torch.einsum('...ij,...jk->...ik', I_in, torch.einsum('...ij,...jk->...ik', P_mat, I_out )) 

    return M_layer


def compute_structure_transfer_matrix_s_pol(n, d, wavelengths, angles, n_env):
    """
    Computes the total transfer matrix for a multilayer structure 
    over all wavelengths and angles in parallel (s-polarization).
    Parameters
    ----------
    n : list[torch.Tensor]
        Refractive indices of shape (num_layers, num_wavelengths), 
        or broadcastable to that shape. E.g. each layer has n[i, :] over wavelengths.
    d : list[torch.Tensor]
        Thicknesses of shape (num_layers, ), or broadcastable to (num_layers, num_wavelengths).
    wavelengths : torch.Tensor
        Wavelengths of shape (num_wavelengths, ), or (num_wavelengths, 1).
    angles : torch.Tensor
        Incidence angles (in degrees).
        Shape: (num_angles,)
    Returns
    -------
    M_total : torch.Tensor
        Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
    """

    # Prepare shapes       
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[0]
    angles_rad = torch.deg2rad(angles) 
    angles_rad = angles_rad.unsqueeze(0).expand(num_wavelengths, num_angles)
    wavelengths = wavelengths.unsqueeze(1).expand(num_wavelengths, num_angles)

    # Initialize total transfer matrix as identity: shape (num_wavelengths, num_angles, 2, 2)
    M_current = torch.eye(2, 2, dtype=torch.complex64).repeat(num_wavelengths, num_angles, 1, 1)

    # calculate x-component of K vector (conserved quantity accros layers)
    n_air = torch.ones((num_wavelengths, num_angles))
    n_env = n_env.unsqueeze(1).expand(num_wavelengths, num_angles)
    angles_rad = torch.arcsin(n_env * torch.sin(angles_rad))
    k_x = n_air * torch.sin(angles_rad)

    for i, n_i in enumerate(n):
        # expand n and d to be broadcastable
        n_air = torch.ones((num_wavelengths, num_angles))
        n_2d = n_i.unsqueeze(1).expand(num_wavelengths, num_angles)
        d_2d= d[i].unsqueeze(0).unsqueeze(1).expand(num_wavelengths, num_angles)

        #retrieve transfer matrix for each layer
        M_layer = single_layer_transfer_matrix_s_pol(n_2d, d_2d, wavelengths, angles_rad, k_x)

        # 5) Update M_current
        M_current = torch.einsum('...ij,...jk->...ik', M_current, M_layer)

    return M_current, k_x

def compute_structure_transfer_matrix_p_pol(n, d, wavelengths, angles, n_env):
    """
    Computes the total transfer matrix for a multilayer structure 
    over all wavelengths and angles in parallel (p-polarization).
    Parameters
    ----------
    n : list[torch.Tensor]
        Refractive indices of shape (num_layers, num_wavelengths), 
        or broadcastable to that shape. E.g. each layer has n[i, :] over wavelengths.
    d : list[torch.Tensor]
        Thicknesses of shape (num_layers, ), or broadcastable to (num_layers, num_wavelengths).
    wavelengths : torch.Tensor
        Wavelengths of shape (num_wavelengths, ), or (num_wavelengths, 1).
    angles : torch.Tensor
        Incidence angles (in degrees).
        Shape: (num_angles,)
    Returns
    -------
    M_total : torch.Tensor
        Overall transfer matrix of shape (num_wavelengths, num_angles, 2, 2).
    """

    # Prepare shapes       
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[0]
    angles_rad = torch.deg2rad(angles) 
    angles_rad = angles_rad.unsqueeze(0).expand(num_wavelengths, num_angles)
    wavelengths = wavelengths.unsqueeze(1).expand(num_wavelengths, num_angles)

    # Initialize total transfer matrix as identity: shape (num_wavelengths, num_angles, 2, 2)
    M_current = torch.eye(2, 2, dtype=torch.complex64).repeat(num_wavelengths, num_angles, 1, 1)

    # calculate x-component of K vector (conserved quantity accros layers)
    n_air = torch.ones((num_wavelengths, num_angles))
    n_env = n_env.unsqueeze(1).expand(num_wavelengths, num_angles)
    angles_rad = torch.arcsin(n_env * torch.sin(angles_rad))
    k_x = n_air * torch.sin(angles_rad)

    for i, n_i in enumerate(n):
        # expand n and d to be broadcastable
        n_air = torch.ones((num_wavelengths, num_angles))
        n_2d = n_i.unsqueeze(1).expand(num_wavelengths, num_angles)
        d_2d= d[i].unsqueeze(0).unsqueeze(1).expand(num_wavelengths, num_angles)

        #retrieve transfer matrix for each layer
        M_layer = single_layer_transfer_matrix_p_pol(n_2d, d_2d, wavelengths, angles_rad, k_x)

        # 5) Update M_current
        M_current = torch.einsum('...ij,...jk->...ik', M_current, M_layer)

    return M_current, k_x


def adding_environment_and_substrate_s_pol(M_current, n_env, n_sub, k_x ):

    n_env = n_env.unsqueeze(1).expand(M_current.shape[:2])
    n_sub = n_sub.unsqueeze(1).expand(M_current.shape[:2])

    n_air = torch.ones(M_current.shape[:2])
    env_int = interface_matrix_s_pol(n_env, n_air, k_x)
    sub_int = interface_matrix_s_pol(n_air, n_sub, k_x)

    M_current = torch.einsum('...ij,...jk->...ik', M_current, sub_int)
    
    M_total = torch.einsum('...ij,...jk->...ik', env_int, M_current)

    return M_total

def adding_environment_and_substrate_p_pol(M_current, n_env, n_sub, k_x ):

    n_env = n_env.unsqueeze(1).expand(M_current.shape[:2])
    n_sub = n_sub.unsqueeze(1).expand(M_current.shape[:2])

    n_air = torch.ones(M_current.shape[:2])
    env_int = interface_matrix_p_pol(n_env, n_air, k_x)
    sub_int = interface_matrix_p_pol(n_air, n_sub, k_x)

    M_current = torch.einsum('...ij,...jk->...ik', M_current, sub_int)
    
    M_total = torch.einsum('...ij,...jk->...ik', env_int, M_current)

    return M_total
