import torch
import time
import copy
from typing import List, Tuple
from torch_tmm.dispersion import Constant_epsilon, Lorentz, BaseDispersion
from torch_tmm.material import BaseMaterial
from torch_tmm.layer import BaseLayer

from tmm_tests.test_dispersion import dispersion_sanity_test
from tmm_tests.test_material import material_sanity_check


def layer_sanity_check(
    layer: "BaseLayer",
    wavelengths: torch.Tensor,
    *,
    verbose: bool = False,
) -> bool:
    """
    Validate a **Layer** object and everything it contains.

    Checks performed
    ----------------
    1.  Layer-level dtype propagation:  
        • `.to(torch.float32)` ⇒ ε, ñ are `complex64`  
        • `.to(torch.float64)` ⇒ ε, ñ are `complex128`
    2.  Thickness (only for `layer_type="coh"`)  
        • scalar, positive, lives on the same dtype / device as the layer
    3.  Material-level dtype propagation (same two stages as above).
    4.  Delegates to :func:`dispersion_sanity_test` for every dispersion
        model inside the material.
    """

    # ------------- helper -------------------------------------------------
    def _report(ok: bool, label: str) -> bool:
        if verbose:
            print(f"[{'✓' if ok else '✗'}] {label}")
        return ok

    stages: List[Tuple[torch.dtype, torch.dtype]] = [
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ]

    # ---------------------------------------------------------------------
    test_layer = copy.deepcopy(layer)          # keep caller untouched
    all_ok = True

    for real_dtype, c_dtype in stages:
        if verbose:
            print(f"\n=== Layer.to({real_dtype}) (expect {c_dtype}) ===")

        test_layer.to(dtype=real_dtype)

        # ---- layer itself ------------------------------------------------
        all_ok &= _report(test_layer.dtype is real_dtype, "layer.dtype")
        all_ok &= _report(
            test_layer.epsilon(wavelengths).dtype is c_dtype,
            "ε dtype",
        )
        all_ok &= _report(
            test_layer.refractive_index(wavelengths).dtype is c_dtype,
            "ñ dtype",
        )

       # ---- thickness (only coherent layers) --------------------------------
        if test_layer.layer_type == "coh":
            t = test_layer.thickness
            all_ok &= _report(t.dtype is real_dtype, "thickness dtype")
            all_ok &= _report(
                t.device == test_layer.material.device,   # <- equality, not identity
                "thickness device",
            )
            all_ok &= _report(
                t.numel() == 1 and (t > 0).all(),
                "thickness positive scalar",
            )

        # ---- material checks -------------------------------------------
        mat = test_layer.material
        if verbose:
            print(f"--- embedded material ({mat.name}) ---")
        all_ok &= _report(mat.dtype is real_dtype, "material.dtype")
        all_ok &= _report(
            mat.epsilon(wavelengths).dtype is c_dtype, "material ε dtype"
        )
        all_ok &= _report(
            mat.refractive_index(wavelengths).dtype is c_dtype, "material ñ dtype"
        )

    if not all_ok:
        return False

    # ------------------------------------------------------------------ dispersion checks
    if verbose:
        print("\n=== Dispersion models inside material ===")

    for idx, disp in enumerate(test_layer.material.dispersion, 1):
        tag = getattr(disp, "name", disp.__class__.__name__)
        if verbose:
            print(f"\n--- Dispersion {idx}: {tag} ---")
        if not dispersion_sanity_test(disp, wavelengths, verbose=verbose):
            if verbose:
                print(f"❌  Layer failed at dispersion {idx}: {tag}")
            return False
            break

    if verbose:
        print("\nAll layer checks passed ✔️")
    return True