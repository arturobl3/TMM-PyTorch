import torch
import time
import copy
from typing import List
from torch_tmm.dispersion import Constant_epsilon, Lorentz, BaseDispersion
from torch_tmm.material import BaseMaterial

from tmm_tests.test_dispersion import dispersion_sanity_test

def material_sanity_check(
    material: BaseMaterial,
    wavelengths: torch.Tensor,
    *,
    verbose: bool = False,
) -> bool:
    """
    Sanity-check the material **and** every dispersion model it contains.

    •  Material-level checks (dtype propagation + ε/ñ dtypes).  
    •  Delegates to :func:`dispersion_sanity_test` for each dispersion.  
    •  Prints the failing dispersion or test when *verbose* is True.

    Returns ``True`` when EVERYTHING passes, otherwise ``False``.
    """

    # ------------------------------------------------------------------ helpers
    def _run(label: str, condition: bool) -> bool:
        ok = bool(condition)
        if verbose:
            mark = "✓" if ok else "✗"
            print(f"[{mark}] {label}")
        return ok

    # ------------------------------------------------------------------ 1. material-level checks
    stages = [
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ]

    test_material = copy.deepcopy(material)

    for real_dtype, c_dtype in stages:
        test_material.to(dtype=real_dtype)

        if verbose:
            print(f"\n=== Material to {real_dtype} (expects {c_dtype}) ===")

        if not _run("material.dtype correct", test_material.dtype is real_dtype):
            return False
        if not _run("ε dtype correct",
                    test_material.epsilon(wavelengths).dtype is c_dtype):
            return False
        if not _run("ñ dtype correct",
                    test_material.refractive_index(wavelengths).dtype is c_dtype):
            return False

    # ------------------------------------------------------------------ 2. dispersion-level checks
    if verbose:
        print("\n=== Checking constituent dispersions ===")

    for idx, disp in enumerate(test_material.dispersion, 1):
        tag = getattr(disp, "name", disp.__class__.__name__)
        if verbose:
            print(f"\n--- Dispersion {idx}: {tag} ---")

        if not dispersion_sanity_test(disp, wavelengths, verbose=verbose):
            if verbose:
                print(f"❌  Material failed at dispersion {idx}: {tag}")
            else:
                print(f"Material check failed (dispersion {idx}: {tag})")
            return False

    return True

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
    

