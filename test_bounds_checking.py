#!/usr/bin/env python3
"""
Simple test script to demonstrate the new bounds checking behavior
of the TabulatedData dispersion class.
"""

import torch
import sys
import os

# Add the TMM-PyTorch root directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_tmm.dispersion import TabulatedData


def test_bounds_checking():
    """Test the new bounds checking behavior."""
    print("TabulatedData Bounds Checking Test")
    print("=" * 40)

    # Create sample data from 500 to 1000 nm
    wavelengths = torch.linspace(500, 1000, 6)
    n_values = torch.linspace(1.4, 1.6, 6) + 0.01j

    print(
        f"Created TabulatedData with range: {wavelengths[0]:.0f} - {wavelengths[-1]:.0f} nm"
    )
    dispersion = TabulatedData(wavelengths, n_values)

    # Test cases
    test_cases = [
        (450.0, "Below range"),
        (750.0, "Within range"),
        (1100.0, "Above range"),
    ]

    print("\nTesting different wavelengths:")
    for wavelength, description in test_cases:
        print(f"\n{description}: λ = {wavelength:.0f} nm")
        try:
            wl_tensor = torch.tensor([wavelength])
            n = dispersion.refractive_index(wl_tensor)
            print(f"  ✓ Success: n = {n[0]:.4f}")
        except ValueError as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 40)
    print("Bounds checking prevents unsafe extrapolation!")


if __name__ == "__main__":
    test_bounds_checking()
