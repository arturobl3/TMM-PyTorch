import torch
import copy
import time
from torch_tmm.dispersion import *


from typing import List, Tuple, Callable

def dispersion_sanity_test(
        dispersion: BaseDispersion,
        wavelengths: torch.Tensor,
        *,
        verbose: bool = False
        ) -> bool:
    """
    Quick consistency check for any `BaseDispersion` subclass.

    1.  Moves the model to float32 ⇒ expects ε, ñ in complex64.
    2.  Moves the model to float64 ⇒ expects ε, ñ in complex128.
    3.  Ensures Im{ε(λ)} and Im{ñ(λ)} are strictly positive
        (causal passive medium).

    Parameters
    ----------
    dispersion : BaseDispersion
        Instance to be verified (will be **modified in-place**).
    wavelengths : torch.Tensor
        Positive wavelengths (same unit used by the model).
    verbose : bool, default False
        If True prints a numbered list and the first failing test.

    Returns
    -------
    bool
        True  → all checks passed  
        False → at least one check failed
    """

    def _msg(ok: bool, label: str) -> str:          # helper for pretty print
        mark = "✓" if ok else "✗"
        return f"[{mark}] {label}"
    
    test_dispersion = copy.deepcopy(dispersion)

    tests: List[Tuple[str, Callable[[], bool]]] = [
        # --- float32 stage -------------------------------------------------
        (
            "move to float32",
            lambda: test_dispersion.to(torch.float32).dtype is torch.float32,
        ),
        (
            "ε dtype = complex64",
            lambda: test_dispersion.epsilon(wavelengths).dtype is torch.complex64,
        ),
        (
            "ñ dtype = complex64",
            lambda: test_dispersion.refractive_index(wavelengths).dtype
            is torch.complex64,
        ),
        # --- float64 stage -------------------------------------------------
        (
            "move to float64",
            lambda: test_dispersion.to(torch.float64).dtype is torch.float64,
        ),
        (
            "ε dtype = complex128",
            lambda: test_dispersion.epsilon(wavelengths).dtype is torch.complex128,
        ),
        (
            "ñ dtype = complex128",
            lambda: test_dispersion.refractive_index(wavelengths).dtype
            is torch.complex128,
        ),
        # --- causality / passivity check -----------------------------------
        (
            "Im{ε} >= 0",
            lambda: torch.all(test_dispersion.epsilon(wavelengths).imag >= 0).item(),
        ),
        (
            "Im{ñ} >= 0",
            lambda: torch.all(test_dispersion.refractive_index(wavelengths).imag >= 0).item(),
        ),
    ]

    all_ok = True
    for i, (label, fn) in enumerate(tests, 1):
        ok = bool(fn())
        all_ok &= ok
        if verbose:
            print(f"{i:02d}. {_msg(ok, label)}")
        if not ok and not verbose:
            break                              # silent, exit early

    # if not verbose:
    #     print("All tests passed ✔️" if all_ok else "❌ Test failed")

    return all_ok


def flat_dispersion_test(epsilon: torch.nn.Parameter,
                         wavelengths : torch.Tensor,
                         dtype: torch.dtype,
                         device: torch.device,
                         verbose: bool = False) -> str:
    
    assert epsilon.ndim == 0, 'Refractive Index must be a scalar tensor' 
    assert wavelengths.ndim == 1, "Wavelengths must be a 1D tensor."

    refractive_index = torch.sqrt(epsilon) 
    #Analytical Method
    refractiveIndex_analytical = torch.ones_like(wavelengths, dtype=dtype, device=device) * refractive_index
    epsilon_analytical = torch.ones_like(wavelengths, dtype=dtype, device=device) * epsilon

    # Dispersion method
    start = time.time()
    flat_e = Constant_epsilon(epsilon, dtype, device)

    n_e = flat_e.refractive_index(wavelengths)
    e_e = flat_e.epsilon(wavelengths)
    end = time.time()
    
    # comparing results

    MSE_n = torch.mean(torch.abs(n_e - refractiveIndex_analytical)**2)
    MSE_e = torch.mean(torch.abs(e_e - epsilon_analytical)**2)

    condition = MSE_n < 1e-8 and MSE_e < 1e-8 


    if verbose:
        return True if condition else False
    else:
        return f"The flat dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_n: {MSE_n}, MSE_e: {MSE_e}"

def lorentz_dispersion_test(A: torch.nn.Parameter,
                            E0: torch.nn.Parameter,
                            C: torch.nn.Parameter,
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
    lorentz = Lorentz(A, E0, C, dtype, device)

    # Compute epsilon and refractive index over the wavelength range
    eps = lorentz.epsilon(wavelengths)             
    n   = lorentz.refractive_index(wavelengths)    
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
        return f"The Lorentz dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_internal: {MSE_internal}, MSE_analytical: {MSE_analytical}"


def Cauchy_dispersion_test(coefficients:List[torch.nn.Parameter],
                            wavelengths: torch.Tensor,
                            dtype: torch.dtype,
                            device: torch.device,
                            verbose: bool = False) -> str:
    """
    Tests the cauchy dispersion model by verifying that n^2(λ) matches ε(λ) 
    to within a specified tolerance (MSE < 1e-8).
    
    Parameters
    ----------
    coefficients: List[float]
        A list of six floats representing the Cauchy coefficients:
                A, B, C for the real part and D, E, F for the imaginary part.
    wavelengths : torch.Tensor
        1D tensor of wavelengths (in m) at which to evaluate the model.
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
    assert len(coefficients) == 6, "Cauchy formula require 6 coefficients."

    start = time.time()

    # Dispersion method
    cauchy = Cauchy(coefficients, dtype, device)

    # Compute epsilon and refractive index over the wavelength range
    eps = cauchy.epsilon(wavelengths)             
    refractive_index  = cauchy.refractive_index(wavelengths)    
    end = time.time()

    #analytical method
    
    A, B, C, D, E, F = coefficients

    n = A + 1e4 * B / wavelengths**2 + 1e9 * C / wavelengths**4
    k = D + 1e4 * E / wavelengths**2 + 1e9 * F / wavelengths**4

    n_analytical = n + 1j * k

    # Evaluate internal consistency: n^2 ~ eps
    MSE_internal = torch.mean(torch.abs(refractive_index**2 - eps)**2)

    MSE_analytical = torch.mean(torch.abs(refractive_index - n_analytical)**2)


    # Define pass/fail condition
    condition = MSE_internal < 1e-8 and MSE_analytical < 1e-8

    if verbose:
        return True if condition else False
    else:
        return f"The Cauchy dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_internal: {MSE_internal}, MSE_analytical: {MSE_analytical}"


def Cauchy_dispersion_test(coefficients:List[torch.nn.Parameter],
                            wavelengths: torch.Tensor,
                            dtype: torch.dtype,
                            device: torch.device,
                            verbose: bool = False) -> str:
    """
    Tests the cauchy dispersion model by verifying that n^2(λ) matches ε(λ) 
    to within a specified tolerance (MSE < 1e-8).
    
    Parameters
    ----------
    coefficients: List[float]
        A list of six floats representing the Cauchy coefficients:
                A, B, C for the real part and D, E, F for the imaginary part.
    wavelengths : torch.Tensor
        1D tensor of wavelengths (in m) at which to evaluate the model.
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
    assert len(coefficients) == 6, "Cauchy formula require 6 coefficients."

    start = time.time()

    # Dispersion method
    cauchy = Cauchy(coefficients, dtype, device)

    # Compute epsilon and refractive index over the wavelength range
    eps = cauchy.epsilon(wavelengths)             
    refractive_index  = cauchy.refractive_index(wavelengths)    
    end = time.time()

    #analytical method
    
    A, B, C, D, E, F = coefficients

    n = A + 1e4 * B / wavelengths**2 + 1e9 * C / wavelengths**4
    k = D + 1e4 * E / wavelengths**2 + 1e9 * F / wavelengths**4

    n_analytical = n + 1j * k

    # Evaluate internal consistency: n^2 ~ eps
    MSE_internal = torch.mean(torch.abs(refractive_index**2 - eps)**2)

    MSE_analytical = torch.mean(torch.abs(refractive_index - n_analytical)**2)


    # Define pass/fail condition
    condition = MSE_internal < 1e-8 and MSE_analytical < 1e-8

    if verbose:
        return True if condition else False
    else:
        return f"The Cauchy dispersion test is:' {'passed' if condition else 'failed'} in {end - start} seconds, MSE_internal: {MSE_internal}, MSE_analytical: {MSE_analytical}"






if __name__ == '__main__':

    ############################ FLAT DISPERSION TEST ###############################
    dtype = torch.complex64
    device = torch.device('cpu')
    epsilon = torch.tensor(2.6+1j, dtype=dtype, device = device)
    epsilon = torch.nn.Parameter(epsilon)
    wavelengths = torch.linspace(400e-9, 800e-9, steps=100, dtype=dtype, device = device)

    print(flat_dispersion_test(epsilon=epsilon, wavelengths=wavelengths, dtype=dtype, device=device))
   
    ########################### LORENTZ DISPERSION TEST ################################
    # Define Lorentz model parameters
    A = 1.0   # Oscillator strength
    E0 = 2.0  # Resonance energy in eV
    C = 0.1   # Damping factor in eV

    # Create a wavelength grid (in nm) as a 1D tensor
    wavelengths = torch.linspace(400e-9, 800e-9, steps=100)  # from 400 nm to 800 nm

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

    ########################### CAUCHY DISPERSION TEST ################################
   # Choose your data type and device
    dtype = torch.complex64
    device = torch.device("cpu")
   
    # Define Cauchy model parameters
    A = torch.tensor(1.0, dtype = dtype, device = device)  
    B = torch.tensor(1.0, dtype = dtype, device = device)  
    C = torch.tensor(0.0, dtype = dtype, device = device)     
    D = torch.tensor(1.0, dtype = dtype, device = device)  
    E = torch.tensor(1.0, dtype = dtype, device = device)  
    F = torch.tensor(0.1, dtype = dtype, device = device)  

    # Create a wavelength grid (in nm) as a 1D tensor
    wavelengths = torch.linspace(400e-9, 800e-9, steps=100)  # from 400 nm to 800 nm

      # or torch.device("cuda") if available

    # Run the test in non-verbose mode (returns a descriptive string)
    test_result_str = Cauchy_dispersion_test([A, B, C, D, E, F], 
                            wavelengths, 
                            dtype, 
                            device, 
                            verbose=False)
    print(test_result_str)
