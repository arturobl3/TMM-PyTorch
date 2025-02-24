import torch
import time
from typing import List
from dispersion import Constant_epsilon, Lorentz, BaseDispersion
from material import BaseMaterial

def base_material_test(dispersions :List[BaseDispersion],
                         wavelengths : torch.Tensor,
                         dtype: torch.dtype,
                         device: torch.device,
                         verbose: bool = False) -> str:
    
    assert wavelengths.ndim == 1, "Wavelengths must be a 1D tensor."

    epsilon_list = [disp.epsilon(wavelengths) for disp in dispersions]
    epsilon_analytical = torch.stack(epsilon_list, dim = 0).sum(dim = 0)

    # Dispersion method
    start = time.time()
    material = BaseMaterial(dispersions, dtype, device)

    refractive_index = material.refractive_index(wavelengths)
    epsilon = material.epsilon(wavelengths)
    end = time.time()
    
    # comparing results

    MSE_internal = torch.mean(torch.abs(refractive_index **2 - epsilon)**2)
    MSE_e = torch.mean(torch.abs(epsilon - epsilon_analytical)**2)

    condition = MSE_internal < 1e-8 and MSE_e < 1e-8 


    if verbose:
        return True if condition else False
    else:
        return f"The base material test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_n: {MSE_internal}, MSE_e: {MSE_e}"
    

if __name__ == '__main__':

    dtype = torch.complex64
    device = torch.device('cpu')

    #constant epsilon
    epsilon_const = torch.tensor(2.6+1j, dtype=dtype, device = device)
    constant_dispersion = Constant_epsilon(epsilon_const=epsilon_const, dtype=dtype, device=device)

    #Lorentz 
    A = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))   # Oscillator strength
    E0 = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype, device=device))   # Resonance energy in eV
    C = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype, device=device))   # Damping factor in eV

    Lorentz_dispersion = Lorentz(A, E0, C, dtype, device)

    ###
    wavelengths = torch.linspace(400e-9, 800e-9, steps=100)

    print(base_material_test([constant_dispersion, Lorentz_dispersion],
                            wavelengths, 
                            dtype, 
                            device, 
                            verbose=False))
    

