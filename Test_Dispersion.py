import torch
import time
from Dispersion import *

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

def lorentz_dispersion_test(A: float,
                            E0: float,
                            C: float,
                            wavelengths: torch.Tensor,
                            dtype: torch.dtype,
                            device: torch.device,
                            verbose: bool = False) -> str:
    """
    Tests the Lorentz dispersion model by verifying that n^2(λ) matches ε(λ) 
    to within a specified tolerance (MSE < 1e-8).
    
    Parameters
    ----------
    A : float
        Oscillator strength (dimensionless).
    E0 : float
        Resonance energy (in eV).
    C : float
        Damping factor (in eV).
    wavelengths : torch.Tensor
        1D tensor of wavelengths (in nm) at which to evaluate the model.
    dtype : torch.dtype
        PyTorch data type (e.g., torch.float32 or torch.complex64).
    device : torch.device
        PyTorch device (e.g., 'cpu' or 'cuda').
    verbose : bool, optional
        If True, return a boolean pass/fail. If False, return a descriptive string.

    Returns
    -------
    str or bool
        If verbose is False, returns a string indicating pass/fail status, elapsed time,
        and MSE. If verbose is True, returns a boolean indicating pass/fail.
    """
    # Basic sanity check
    assert wavelengths.ndim == 1, "Wavelengths must be a 1D tensor."

    start = time.time()

    # Dispersion method
    lorentz = Lorentz(A, E0, C, wavelengths, dtype, device)

    # Compute epsilon and refractive index over the wavelength range
    eps = lorentz.getEpsilon(wavelengths)             
    n   = lorentz.getRefractiveIndex(wavelengths)    
    end = time.time()

    #analytical method
    plank_constant = torch.tensor(6.62607015e-34, dtype=dtype, device = device)
    c_constant = torch.tensor(299792458, dtype=dtype, device = device)
    e_constant = torch.tensor(1.60217663e-19, dtype=dtype, device = device)

    E = (plank_constant * c_constant / e_constant) / (wavelengths)
    
    eps_analytical = (A * E0) / (E0**2 - E**2 - 1j * C * E)

    # Evaluate internal consistency: n^2 ~ eps
    MSE_internal = torch.mean(torch.abs(n**2 - eps)**2)

    MSE_analytical = torch.mean(torch.abs(eps - eps_analytical)**2)


    # Define pass/fail condition
    condition = MSE_internal < 1e-8 and MSE_analytical < 1e-8

    if verbose:
        return True if condition else False
    else:
        return f"The flat dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_internal: {MSE_internal}, MSE_analytical: {MSE_analytical}"



if __name__ == '__main__':

    ############################ FLAT DISPERSION TEST ###############################
    dtype = torch.complex64
    device = torch.device('cpu')
    n = torch.tensor(2.6+1j, dtype=dtype, device = device )
    num_wavelength = 30

    print(flat_dispersion_test(refractiveIndex=n, num_wavelenght = num_wavelength, dtype=dtype, device=device))
   
    ########################### LORENTZ DISPERSION TEST ################################
    # Define Lorentz model parameters
    A = 1.0   # Oscillator strength
    E0 = 2.0  # Resonance energy in eV
    C = 0.1   # Damping factor in eV

    # Create a wavelength grid (in nm) as a 1D tensor
    wavelengths = torch.linspace(400, 800, steps=100)  # from 400 nm to 800 nm

    # Choose your data type and device
    dtype = torch.complex64
    device = torch.device("cpu")  # or torch.device("cuda") if available

    # Run the test in non-verbose mode (returns a descriptive string)
    test_result_str = lorentz_dispersion_test(A, E0, C, 
                                              wavelengths, 
                                              dtype, 
                                              device, 
                                              verbose=False)
    print(test_result_str)
