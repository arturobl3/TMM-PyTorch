import torch
import numpy as np
import time
from model import Model
from dispersion import Constant_epsilon
from material import BaseMaterial
from layer import BaseLayer
from typing import List


def model_test_single_layer(wavelengths: torch.Tensor,
                        angles: torch.Tensor, 
                        structure: List[BaseLayer],
                        env: BaseLayer,
                        subs: BaseLayer,
                        dtype: torch.dtype = torch.complex64,
                        device: torch.device = torch.device('cpu'),
                        verbose: bool = False) -> str:
    '''
    Test the Transfer matrix for a single layer.

    Parameters
    ----------
    wavelengths : torch.Tensor
        Wavelengths to test.
    angles : torch.Tensor
        Angles to test.
    structure: List[BaseLayer]
        List of layers of the structure
    n_env : torch.Tensor
        Refractive index of the environment    
    n_subs : torch.Tensor
        Refractive index of the substrate   
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.complex64
    device : torch.device, optional
        Device to run the tensors on, by default torch.device('cpu')
    '''
    #Transfer input data to the correct device and data type
    wavelengths = wavelengths.to(dtype).to(device)
    angles = angles.to(dtype).to(device)
    n = structure[0].material.refractive_index(wavelengths).to(dtype).to(device)
    d = structure[0].thickness.to(dtype).to(device)
    n_env = env.material.refractive_index(wavelengths).to(dtype).to(device)
    n_subs = subs.material.refractive_index(wavelengths).to(dtype).to(device)

    assert len(structure) == 1, 'single layer structure'
    assert wavelengths.ndim == 1, 'Wavelengths must be a 1D tensor'
    assert angles.ndim == 1, 'Angles must be a 1D tensor'
    assert n.ndim == 1, 'Refractive index must be a 1D tensor'
    assert d.ndim == 0, 'Thickness must be a scalar tensor'
    assert n_env.ndim == 1, 'Refractive index of the environment must be a 1D tensor'
    assert n_subs.ndim == 1, 'Refractive index of the substrate must be a 1D tensor'
    assert wavelengths.shape[0] == n_env.shape[0], 'Wavelengths and refractive index of the environment must have the same length'
    assert wavelengths.shape[0] == n_subs.shape[0], 'Wavelengths and refractive index of the substrate must have the same length'
    assert wavelengths.shape[0] == n.shape[0], 'Wavelengths and refractive index must have the same length'

    #T matrix method
    start = time.time()
    model = Model(env, structure, subs, dtype, device)
    OpticalProperties = model.evaluate(wavelengths, angles)
    
    
    R_s, R_p = OpticalProperties.reflection()
    T_s, T_p = OpticalProperties.transmission()
    end = time.time()

    #Analytical method
    angles_rad = torch.deg2rad(angles.to(torch.float64)).to(dtype)
    nx = n_env[:, None] * torch.sin(angles_rad[None, :])
    n1z = torch.sqrt(n_env[:, None]**2 - nx**2)
    n2z = torch.sqrt(n[:,None]**2 - nx**2)
    n3z = torch.sqrt(n_subs[:, None]**2 - nx**2)
    beta = 2*np.pi/wavelengths[:, None]*n2z*d

    ## s polarization
    r12 = (n1z - n2z)/(n1z + n2z)
    r23 = (n2z - n3z)/(n2z + n3z)
    t12 = 2*n1z/(n1z + n2z)
    t23 = 2*n2z/(n2z + n3z)

    r_analytical = (r12 + r23*torch.exp(2j*beta))/(1 + r12*r23*torch.exp(2j*beta))
    t_analytical = t12*t23*torch.exp(1j*beta)/(1 + r12*r23*torch.exp(2j*beta))

    R_analytical = torch.abs(r_analytical)**2
    T_analytical = torch.abs(t_analytical)**2 * torch.real(n3z/n1z)

    MSE_R_s = torch.mean(torch.abs(R_s - R_analytical)**2)
    MSE_T_s = torch.mean(torch.abs(T_s - T_analytical)**2)

    ## p polarization
    r12 = (n[:,None]**2*n1z - n_env[:,None]**2*n2z)/(n[:,None]**2*n1z + n_env[:,None]**2*n2z)
    r23 = (n_subs[:,None]**2*n2z - n[:,None]**2*n3z)/(n_subs[:,None]**2*n2z + n[:,None]**2*n3z)
    t12 = 2*n[:,None]**2*n1z/(n[:,None]**2*n1z + n_env[:,None]**2*n2z)
    t23 = 2*n_subs[:,None]**2*n2z/(n_subs[:,None]**2*n2z + n[:,None]**2*n3z)

    r_analytical = (r12 + r23*torch.exp(2j*beta))/(1 + r12*r23*torch.exp(2j*beta))
    t_analytical = t12*t23*torch.exp(1j*beta)/(1 + r12*r23*torch.exp(2j*beta))

    R_analytical = torch.abs(r_analytical)**2
    T_analytical = torch.abs(t_analytical)**2 * torch.real(n3z/n1z)

    MSE_R_p = torch.mean(torch.abs(R_p - R_analytical)**2)
    MSE_T_p = torch.mean(torch.abs(T_p - T_analytical)**2)

    condition = MSE_R_s < 1e-8 and MSE_T_s < 1e-8 and MSE_R_p < 1e-8 and MSE_T_p < 1e-8

    if verbose:
        return True if condition else False
    else:
        return f"The model_test_single_layer {'passed' if condition else 'failed'} in {end - start} seconds,\n s-polarization : MSE_R_s: {MSE_R_s}, MSE_T_s: {MSE_T_s},\n p-polarization: MSE_R_p: {MSE_R_p}, MSE_T_p: {MSE_T_p}" 


def model_test_double_layer(wavelengths: torch.Tensor,
                        angles: torch.Tensor, 
                        structure: List[BaseLayer],
                        env: BaseLayer,
                        subs: BaseLayer,
                        dtype: torch.dtype = torch.complex64,
                        device: torch.device = torch.device('cpu'),
                        verbose: bool = False) -> str:
    '''
    Test the Transfer matrix for a single layer.

    Parameters
    ----------
    wavelengths : torch.Tensor
        Wavelengths to test.
    angles : torch.Tensor
        Angles to test.
    structure: List[BaseLayer]
        List of layers of the structure
    n_env : torch.Tensor
        Refractive index of the environment    
    n_subs : torch.Tensor
        Refractive index of the substrate   
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.complex64
    device : torch.device, optional
        Device to run the tensors on, by default torch.device('cpu')
    '''
    #Transfer input data to the correct device and data type
    wavelengths = wavelengths.to(dtype).to(device)
    angles = angles.to(dtype).to(device)
    n1 = structure[0].material.refractive_index(wavelengths).to(dtype).to(device)
    n2 = structure[1].material.refractive_index(wavelengths).to(dtype).to(device)
    d1 = structure[0].thickness.to(dtype).to(device)
    d2 = structure[1].thickness.to(dtype).to(device)
    n_env = env.material.refractive_index(wavelengths).to(dtype).to(device)
    n_subs = subs.material.refractive_index(wavelengths).to(dtype).to(device)

    assert len(structure) == 2, 'double layer structure'
    assert wavelengths.ndim == 1, 'Wavelengths must be a 1D tensor'
    assert angles.ndim == 1, 'Angles must be a 1D tensor'
    assert n1.ndim == 1, 'Refractive index must be a 1D tensor'
    assert d1.ndim == 0, 'Thickness must be a scalar tensor'
    assert n2.ndim == 1, 'Refractive index must be a 1D tensor'
    assert d2.ndim == 0, 'Thickness must be a scalar tensor'
    assert n_env.ndim == 1, 'Refractive index of the environment must be a 1D tensor'
    assert n_subs.ndim == 1, 'Refractive index of the substrate must be a 1D tensor'
    assert wavelengths.shape[0] == n_env.shape[0], 'Wavelengths and refractive index of the environment must have the same length'
    assert wavelengths.shape[0] == n_subs.shape[0], 'Wavelengths and refractive index of the substrate must have the same length'
    assert wavelengths.shape[0] == n1.shape[0], 'Wavelengths and refractive index must have the same length'
    assert wavelengths.shape[0] == n2.shape[0], 'Wavelengths and refractive index must have the same length'

    #T matrix method
    start = time.time()
    model = Model(env, structure, subs, dtype, device)
    OpticalProperties = model.evaluate(wavelengths, angles)
    
    
    R_s, R_p = OpticalProperties.reflection()
    T_s, T_p = OpticalProperties.transmission()
    end = time.time()

    #Analytical method
    angles_rad = torch.deg2rad(angles.to(torch.float64)).to(dtype)
    nx = n_env[:, None] * torch.sin(angles_rad[None, :])
    n1z = torch.sqrt(n_env[:, None]**2 - nx**2)
    n2z = torch.sqrt(n1[:,None]**2 - nx**2)
    n3z = torch.sqrt(n2[:,None]**2 - nx**2)
    n4z = torch.sqrt(n_subs[:, None]**2 - nx**2)
    beta1 = 2*np.pi/wavelengths[:, None]*n2z*d1
    beta2 = 2*np.pi/wavelengths[:, None]*n3z*d2

    ## s polarization
    r12 = (n1z - n2z)/(n1z + n2z)
    r23 = (n2z - n3z)/(n2z + n3z)
    r34 = (n3z - n4z)/(n3z +n4z)
    t12 = 2*n1z/(n1z + n2z)
    t23 = 2*n2z/(n2z + n3z)
    t34 = 2*n3z/(n3z+n4z)

    r234 = (r23 + r34*torch.exp(2j*beta2))/(1+r23*r34*torch.exp(2j*beta2))
    t234 = t23*t34*torch.exp(1j*beta2)/(1+r23*r34*torch.exp(2j*beta2))

    r_analytical = (r12 + r234*torch.exp(2j*beta1))/(1 + r12*r234*torch.exp(2j*beta1))
    t_analytical = t12*t234*torch.exp(1j*beta1)/(1 + r12*r234*torch.exp(2j*beta1))

    R_analytical = torch.abs(r_analytical)**2
    T_analytical = torch.abs(t_analytical)**2 * torch.real(n4z/n1z)

    MSE_R_s = torch.mean(torch.abs(R_s - R_analytical)**2)
    MSE_T_s = torch.mean(torch.abs(T_s - T_analytical)**2)

    ## p polarization
    r12 = (n1[:,None]**2*n1z - n_env[:,None]**2*n2z)/(n1[:,None]**2*n1z + n_env[:,None]**2*n2z)
    r23 = (n2[:,None]**2*n2z - n1[:,None]**2*n3z)/(n2[:,None]**2*n2z + n1[:,None]**2*n3z)
    r34 = (n_subs[:,None]**2*n3z - n2[:,None]**2*n4z)/(n_subs[:,None]**2*n3z + n2[:,None]**2*n4z)
    t12 = 2*n1[:,None]**2*n1z/(n1[:,None]**2*n1z + n_env[:,None]**2*n2z)
    t23 = 2*n2[:,None]**2*n2z/(n2[:,None]**2*n2z + n1[:,None]**2*n3z)
    t34 =2*n_subs[:,None]**2*n3z/(n_subs[:,None]**2*n3z + n2[:,None]**2*n4z)

    r234 = (r23 + r34*torch.exp(2j*beta2))/(1 + r23*r34*torch.exp(2j*beta2))
    t234 = t23*t34*torch.exp(1j*beta2)/(1 + r23*r34*torch.exp(2j*beta2))

    r_analytical = (r12 + r234*torch.exp(2j*beta1))/(1 + r12*r234*torch.exp(2j*beta1))
    t_analytical = t12*t234*torch.exp(1j*beta1)/(1 + r12*r234*torch.exp(2j*beta1))

    R_analytical = torch.abs(r_analytical)**2
    T_analytical = torch.abs(t_analytical)**2 * torch.real(n4z/n1z)

    MSE_R_p = torch.mean(torch.abs(R_p - R_analytical)**2)
    MSE_T_p = torch.mean(torch.abs(T_p - T_analytical)**2)

    condition = MSE_R_s < 1e-8 and MSE_T_s < 1e-8 and MSE_R_p < 1e-8 and MSE_T_p < 1e-8

    if verbose:
        return True if condition else False
    else:
        return f"The model_test_double_layer {'passed' if condition else 'failed'} in {end - start} seconds,\n s-polarization : MSE_R_s: {MSE_R_s}, MSE_T_s: {MSE_T_s},\n p-polarization: MSE_R_p: {MSE_R_p}, MSE_T_p: {MSE_T_p}" 



if __name__ == '__main__':

    dtype=torch.complex64
    device=torch.device('cpu')

    wavelengths = torch.linspace(400, 800, 401)
    angles = torch.linspace(0, 89, 90)

    env = BaseLayer(BaseMaterial([Constant_epsilon(1, dtype, device)], dtype, device), thickness=0, LayerType='env')
    subs = BaseLayer(BaseMaterial([Constant_epsilon(6, dtype, device)], dtype, device),thickness= 0, LayerType='subs')

    n_layer = Constant_epsilon(3+0.2j, dtype=dtype, device=device)
    d_layer = torch.tensor(30,dtype=dtype, device=device)
    layer = BaseLayer(BaseMaterial([n_layer], dtype, device), thickness=d_layer, LayerType='coh')

    print(model_test_single_layer(wavelengths, angles, [layer], env, subs,
                    dtype=dtype, device=device))
    

    env = BaseLayer(BaseMaterial([Constant_epsilon(1, dtype, device)], dtype, device), thickness=0, LayerType='env')
    subs = BaseLayer(BaseMaterial([Constant_epsilon(6, dtype, device)], dtype, device),thickness= 0, LayerType='subs')

    n_1 = Constant_epsilon(3+0.2j, dtype=dtype, device=device)
    n_2 = Constant_epsilon(1+0.5j, dtype=dtype, device=device)
    d_1 = torch.tensor(30,dtype=dtype, device=device)
    d_2 = torch.tensor(15,dtype=dtype, device=device)
    layer_1 = BaseLayer(BaseMaterial([n_1], dtype, device), thickness=d_1, LayerType='coh')
    layer_2 = BaseLayer(BaseMaterial([n_2], dtype, device), thickness=d_2, LayerType='coh')

    print(model_test_double_layer(wavelengths, angles, [layer_1, layer_2], env, subs,
                    dtype=dtype, device=device))

