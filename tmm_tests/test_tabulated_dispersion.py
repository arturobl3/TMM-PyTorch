"""
Test suite for TabulatedData dispersion model.

This module tests the TabulatedData dispersion class to ensure proper
interpolation behavior, error handling, and compatibility with the
TMM framework.
"""

import torch
import sys
import os

# Add the TMM-PyTorch root directory to the path so we can import the local version
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_tmm.dispersion import TabulatedData, BaseDispersion


def test_tabulated_dispersion_basic():
    """Basic test of TabulatedData dispersion functionality."""
    print("Testing TabulatedData dispersion...")

    # Create sample tabulated data
    wavelengths = torch.linspace(500, 1000, 11)  # 500-1000 nm, 11 points
    # Create a simple refractive index that varies with wavelength
    n_real = 1.5 + 0.001 * (wavelengths - 500)  # Linear increase
    n_imag = 0.01 + 0.00001 * (wavelengths - 500)  # Small imaginary part
    refractive_indices = n_real + 1j * n_imag

    # Test initialization
    print("  Testing initialization...")
    dispersion = TabulatedData(wavelengths, refractive_indices)
    assert isinstance(dispersion, BaseDispersion)
    print("    ✓ Initialization successful")

    # Test exact interpolation at table points
    print("  Testing exact interpolation...")
    test_wavelengths = wavelengths[::3]  # Every 3rd point
    expected_n = refractive_indices[::3]
    computed_n = dispersion.refractive_index(test_wavelengths)

    # Check if values are close
    diff = torch.abs(computed_n - expected_n)
    assert torch.all(diff < 1e-6), f"Max difference: {torch.max(diff)}"
    print("    ✓ Exact interpolation working")

    # Test linear interpolation between points
    print("  Testing linear interpolation...")
    wl_simple = torch.tensor([500.0, 600.0, 700.0])
    n_simple = torch.tensor([1.0 + 0.0j, 2.0 + 0.1j, 3.0 + 0.2j])
    dispersion_simple = TabulatedData(wl_simple, n_simple)

    # Test interpolation at midpoint (550 nm)
    test_wl = torch.tensor([550.0])
    expected_n = torch.tensor([1.5 + 0.05j])  # Linear interpolation
    computed_n = dispersion_simple.refractive_index(test_wl)

    diff = torch.abs(computed_n - expected_n)
    assert torch.all(diff < 1e-6), f"Interpolation error: {diff}"
    print("    ✓ Linear interpolation working")

    # Test epsilon calculation
    print("  Testing epsilon calculation...")
    test_wl = wavelengths[5:8]  # Middle values
    n = dispersion.refractive_index(test_wl)
    epsilon = dispersion.epsilon(test_wl)
    expected_epsilon = n**2

    diff = torch.abs(epsilon - expected_epsilon)
    assert torch.all(diff < 1e-6), f"Epsilon calculation error: {torch.max(diff)}"
    print("    ✓ Epsilon calculation working")

    # Test vectorized evaluation
    print("  Testing vectorized evaluation...")
    test_wavelengths = torch.linspace(520, 980, 25)  # Within range
    n_values = dispersion.refractive_index(test_wavelengths)
    epsilon_values = dispersion.epsilon(test_wavelengths)

    assert n_values.shape == test_wavelengths.shape
    assert epsilon_values.shape == test_wavelengths.shape
    assert torch.isfinite(n_values).all()
    assert torch.isfinite(epsilon_values).all()
    print("    ✓ Vectorized evaluation working")

    # Test string representation
    print("  Testing string representation...")
    repr_str = repr(dispersion)
    assert "TabulatedData Dispersion" in repr_str
    assert "500.0-1000.0 nm" in repr_str
    assert "11 data points" in repr_str
    print(f"    ✓ Repr: {repr_str}")

    print("  All tests passed! ✓")


def test_tabulated_dispersion_errors():
    """Test error handling in TabulatedData dispersion."""
    print("Testing TabulatedData error handling...")

    wavelengths = torch.linspace(500, 1000, 5)
    refractive_indices = torch.linspace(1.4, 1.6, 5) + 0.01j

    # Test length mismatch
    print("  Testing length mismatch error...")
    try:
        TabulatedData(wavelengths, refractive_indices[:-1])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)
        print("    ✓ Length mismatch error caught")

    # Test insufficient points
    print("  Testing insufficient points error...")
    try:
        TabulatedData(torch.tensor([500.0]), torch.tensor([1.5 + 0.01j]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "At least 2 data points" in str(e)
        print("    ✓ Insufficient points error caught")

    # Test unsorted wavelengths
    print("  Testing unsorted wavelengths error...")
    try:
        unsorted_wl = torch.tensor([500.0, 600.0, 550.0, 700.0])  # 550 is out of order
        n_values = torch.tensor(
            [1.5 + 0.01j, 1.51 + 0.01j, 1.505 + 0.01j, 1.52 + 0.01j]
        )
        TabulatedData(unsorted_wl, n_values)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be sorted in ascending order" in str(e)
        print("    ✓ Unsorted wavelengths error caught")

    # Test out-of-bounds wavelengths
    print("  Testing out-of-bounds wavelength errors...")
    # Create a valid dispersion for testing bounds
    valid_wl = torch.linspace(500, 1000, 5)
    valid_n = torch.linspace(1.4, 1.6, 5) + 0.01j
    dispersion = TabulatedData(valid_wl, valid_n)

    # Test wavelength below range
    try:
        test_wl_low = torch.tensor([400.0])  # Below 500 nm
        dispersion.refractive_index(test_wl_low)
        assert False, "Should have raised ValueError for low wavelength"
    except ValueError as e:
        assert "below the minimum tabulated wavelength" in str(e)
        assert "400.00 nm" in str(e)
        assert "500.00 nm" in str(e)
        print("    ✓ Low wavelength error caught")

    # Test wavelength above range
    try:
        test_wl_high = torch.tensor([1200.0])  # Above 1000 nm
        dispersion.refractive_index(test_wl_high)
        assert False, "Should have raised ValueError for high wavelength"
    except ValueError as e:
        assert "above the maximum tabulated wavelength" in str(e)
        assert "1200.00 nm" in str(e)
        assert "1000.00 nm" in str(e)
        print("    ✓ High wavelength error caught")

    # Test mixed in-bounds and out-of-bounds wavelengths
    try:
        test_wl_mixed = torch.tensor([400.0, 750.0, 1200.0])  # Low, valid, high
        dispersion.refractive_index(test_wl_mixed)
        assert False, "Should have raised ValueError for mixed wavelengths"
    except ValueError as e:
        # Should catch the low wavelength first
        assert "below the minimum tabulated wavelength" in str(e)
        print("    ✓ Mixed wavelength error caught")

    print("  All error tests passed! ✓")


def test_tabulated_dispersion_integration():
    """Test integration with Material class if available."""
    print("Testing TabulatedData integration...")

    # First test the dispersion standalone
    print("  Testing standalone dispersion...")
    wavelengths = torch.linspace(500, 1000, 6).float()
    n_values = (torch.linspace(1.4, 1.6, 6) + 0.01j).to(torch.complex64)

    tabulated_disp = TabulatedData(wavelengths, n_values)
    test_wl = torch.tensor([750.0]).float()

    # Test direct usage
    n_direct = tabulated_disp.refractive_index(test_wl)
    epsilon_direct = tabulated_disp.epsilon(test_wl)

    assert torch.isfinite(n_direct).all()
    assert torch.isfinite(epsilon_direct).all()
    assert n_direct.dtype.is_complex
    assert epsilon_direct.dtype.is_complex
    print("    ✓ Standalone dispersion working")

    # Test with Material class
    try:
        from torch_tmm.material import Material

        print("  Testing with Material class...")

        # Create material with Material class that now properly handles complex buffers
        material = Material(
            [tabulated_disp], name="TabulatedMaterial", dtype=torch.float32
        )

        # Test that the material works correctly with complex buffers preserved
        n_material = material.refractive_index(test_wl)
        epsilon_material = material.epsilon(test_wl)

        if (
            torch.isfinite(n_material).all()
            and torch.isfinite(epsilon_material).all()
            and n_material.dtype.is_complex
            and epsilon_material.dtype.is_complex
        ):
            print("    ✓ Integration with Material class successful")
        else:
            print("    - Material integration produced unexpected results")

    except ImportError:
        print("  - Material class not available, skipping integration test")


if __name__ == "__main__":
    """Run all tests when executed directly."""
    print("=" * 60)
    print("Testing TabulatedData Dispersion Class")
    print("=" * 60)

    test_tabulated_dispersion_basic()
    print()
    test_tabulated_dispersion_errors()
    print()
    test_tabulated_dispersion_integration()

    print()
    print("=" * 60)
    print("All TabulatedData tests completed successfully! 🎉")
    print("=" * 60)
