import torch
import numpy as np
import time
from Dispersion import flatEpsilon

def flat_dispersion_test(refractiveIndex: torch.Tensor,
                         num_wavelenght : int,
                         dtype: torch.dtype,
                         device: torch.device,
                         verbose: bool = False) -> str:
    
    assert refractiveIndex.ndim == 0, 'Refractive Index must be a scalar tensor' 

    epsilon = refractiveIndex ** 2
    #Analytical Method
    refractiveIndex_analytical = torch.ones(num_wavelenght, dtype=dtype, device=device) * refractiveIndex
    epsilon_analytical = torch.ones(num_wavelenght, dtype=dtype, device=device) * epsilon

    # Dispersion method
    start = time.time()
    flat_e = flatEpsilon(0, num_wavelenght, dtype, device)
    flat_e.updateParams(epsilon)



    n_e = flat_e.getRefractiveIndex()
    e_e = flat_e.getEpsilon()
    end = time.time()
    
    # comparing results

    MSE_n = torch.mean(torch.abs(n_e - refractiveIndex_analytical)**2)
    MSE_e = torch.mean(torch.abs(e_e - epsilon_analytical)**2)

    condition = MSE_n < 1e-8 and MSE_e < 1e-8 


    if verbose:
        return True if condition else False
    else:
        return f"The flat dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_n: {MSE_n}, MSE_e: {MSE_e}"



if __name__ == '__main__':
    dtype = torch.complex64
    device = torch.device('cpu')
    n = torch.tensor(2.6+1j, dtype=dtype, device = device )
    num_wavelength = 30

    print(flat_dispersion_test(refractiveIndex=n, num_wavelenght = num_wavelength, dtype=dtype, device=device))