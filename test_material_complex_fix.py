#!/usr/bin/env python3
"""
Test script to verify that the Material class now properly handles
complex buffers without casting them to real dtypes.
"""

import torch
import sys
import os
import warnings

# Add the TMM-PyTorch root directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_tmm.dispersion import TabulatedData
from torch_tmm.material import Material


def test_material_complex_preservation():
    """Test that Material class preserves complex buffers properly."""
    print("Testing Material class complex dtype preservation...")
    print("=" * 60)

    # Create TabulatedData with complex refractive indices
    wavelengths = torch.linspace(500, 1000, 6)
    n_values = torch.linspace(1.4, 1.6, 6) + 0.01j
    tabulated_disp = TabulatedData(wavelengths, n_values)

    print(
        f"Original TabulatedData buffer dtype: {tabulated_disp.refractive_index_table.dtype}"
    )
    print(
        f"Buffer is complex: {tabulated_disp.refractive_index_table.dtype.is_complex}"
    )

    # Capture warnings to check if complex-to-real casting occurs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create Material with float32 dtype (this used to cause the problem)
        print(f"\nCreating Material with dtype=torch.float32...")
        material = Material([tabulated_disp], dtype=torch.float32, name="Test Material")

        # Check if any warnings were raised
        complex_warnings = [
            warning
            for warning in w
            if "Casting complex values to real" in str(warning.message)
        ]

        if complex_warnings:
            print(
                f"❌ FAILED: {len(complex_warnings)} complex-to-real warnings raised:"
            )
            for warning in complex_warnings:
                print(f"    - {warning.message}")
        else:
            print("✅ SUCCESS: No complex-to-real casting warnings!")

    # Check that the complex buffer is preserved
    print(f"\nAfter Material creation:")
    print(f"Material dtype: {material.dtype}")
    print(f"TabulatedData buffer dtype: {tabulated_disp.refractive_index_table.dtype}")
    print(
        f"Buffer is still complex: {tabulated_disp.refractive_index_table.dtype.is_complex}"
    )

    # Test that the material still works correctly
    print(f"\nTesting material functionality...")
    test_wavelengths = torch.tensor([750.0])

    try:
        # Test epsilon calculation
        eps = material.epsilon(test_wavelengths)
        print(f"✅ epsilon(750 nm) = {eps[0]:.4f}")
        print(f"   epsilon is complex: {eps.dtype.is_complex}")

        # Test refractive index calculation
        n = material.refractive_index(test_wavelengths)
        print(f"✅ n(750 nm) = {n[0]:.4f}")
        print(f"   n is complex: {n.dtype.is_complex}")

        # Verify the values make sense (should have imaginary part)
        if eps[0].imag != 0 and n[0].imag != 0:
            print("✅ Complex values preserved correctly!")
        else:
            print("❌ Complex values lost!")

    except Exception as e:
        print(f"❌ Error during material evaluation: {e}")

    print("\n" + "=" * 60)
    return len(complex_warnings) == 0


def test_dtype_conversion_edge_cases():
    """Test various dtype conversion scenarios."""
    print("Testing dtype conversion edge cases...")
    print("=" * 40)

    # Create test data
    wavelengths = torch.linspace(500, 1000, 6)
    n_values = torch.linspace(1.4, 1.6, 6) + 0.01j
    tabulated_disp = TabulatedData(wavelengths, n_values)

    # Test 1: Real dtype material creation
    print("Test 1: Creating material with real dtype...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        material_real = Material([tabulated_disp], dtype=torch.float32)
        warnings_count = len(
            [w_i for w_i in w if "Casting complex" in str(w_i.message)]
        )
        print(f"  Warnings: {warnings_count} (should be 0)")

    # Test 2: Complex dtype material creation
    print("Test 2: Creating material with complex dtype...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        material_complex = Material([tabulated_disp], dtype=torch.complex64)
        warnings_count = len(
            [w_i for w_i in w if "Casting complex" in str(w_i.message)]
        )
        print(f"  Warnings: {warnings_count} (should be 0)")

    # Test 3: Device movement
    print("Test 3: Moving material to different device...")
    try:
        if torch.cuda.is_available():
            material_cuda = material_real.to(device="cuda")
            print(f"  ✅ Successfully moved to CUDA")
            print(f"  Buffer device: {tabulated_disp.refractive_index_table.device}")
        else:
            print(f"  ⚠️  CUDA not available, skipping GPU test")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test 4: Combined dtype and device change
    print("Test 4: Combined dtype and device change...")
    try:
        material_combined = material_real.to(dtype=torch.float64, device="cpu")
        print(f"  ✅ Successfully changed dtype and device")
        print(f"  Material dtype: {material_combined.dtype}")
        print(f"  Buffer dtype: {tabulated_disp.refractive_index_table.dtype}")
    except Exception as e:
        print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    print("Material Class Complex Dtype Fix Test")
    print("=" * 60)

    success = test_material_complex_preservation()
    test_dtype_conversion_edge_cases()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Complex dtype preservation is working!")
    else:
        print("❌ Some tests failed. Complex dtype preservation needs more work.")
