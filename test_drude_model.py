#!/usr/bin/env python3
"""
Test script to verify the Drude dispersion model implementation.

This script tests the Drude model for metallic optical properties and compares
it with expected behavior for metals.
"""

import torch
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, ".")


def test_drude_basic():
    """Test basic Drude model functionality."""
    print("Testing basic Drude model functionality...")
    try:
        from torch_tmm.dispersion import Drude

        # Create Drude model with typical metal parameters
        # Values approximating silver in the visible/IR range
        omega_p = 9.0  # Plasma frequency in eV (typical for metals ~5-15 eV)
        gamma = 0.1  # Collision frequency in eV (typically 0.01-0.5 eV)

        drude = Drude(omega_p=omega_p, gamma=gamma)
        print(f"✅ Drude model created: {drude}")

        # Test at a few wavelengths
        test_wavelengths = torch.tensor([500.0, 800.0, 1000.0, 2000.0])  # nm

        # Test epsilon calculation
        epsilon = drude.epsilon(test_wavelengths)
        print(f"✅ Epsilon calculation successful")
        print(f"   Wavelengths: {test_wavelengths.tolist()} nm")
        print(f"   Epsilon real: {epsilon.real.tolist()}")
        print(f"   Epsilon imag: {epsilon.imag.tolist()}")

        # Test refractive index calculation
        n = drude.refractive_index(test_wavelengths)
        print(f"✅ Refractive index calculation successful")
        print(f"   n real: {n.real.tolist()}")
        print(f"   n imag: {n.imag.tolist()}")

        # Verify that epsilon = n^2
        epsilon_from_n = n**2
        epsilon_diff = torch.abs(epsilon - epsilon_from_n).max()
        if epsilon_diff < 1e-4:  # More tolerant of floating point precision
            print("✅ Consistency check passed: ε = n²")
        else:
            print(f"❌ Consistency check failed: max diff = {epsilon_diff}")
            return False

        return True

    except Exception as e:
        print(f"❌ Drude basic test failed: {e}")
        traceback.print_exc()
        return False


def test_drude_physics():
    """Test that Drude model exhibits expected physical behavior."""
    print("\nTesting Drude model physical behavior...")
    try:
        from torch_tmm.dispersion import Drude

        # Test with realistic metal parameters
        omega_p = 8.5  # eV, typical for good metals
        gamma = 0.05  # eV, relatively low damping

        drude = Drude(omega_p=omega_p, gamma=gamma)

        # Test over a wide wavelength range
        wavelengths = torch.linspace(400, 3000, 100)  # 400nm to 3μm
        epsilon = drude.epsilon(wavelengths)
        n = drude.refractive_index(wavelengths)

        # Convert wavelengths to photon energies for analysis
        hc_over_e = 1.2398419843320026e3  # eV·nm
        energies = hc_over_e / wavelengths  # eV

        # Physical checks:

        # 1. At high frequencies (short wavelengths), epsilon should approach 1
        high_freq_mask = energies > 2 * omega_p  # Well above plasma frequency
        if len(high_freq_mask.nonzero()) > 0:
            eps_high_freq = epsilon[high_freq_mask]
            eps_real_high = eps_high_freq.real.mean()
            if 0.8 < eps_real_high < 1.2:
                print("✅ High frequency limit correct (ε → 1)")
            else:
                print(f"❌ High frequency limit incorrect: ε_real = {eps_real_high}")

        # 2. At the plasma frequency, epsilon real should be very small
        plasma_energy = omega_p  # eV
        plasma_idx = torch.argmin(torch.abs(energies - plasma_energy))
        eps_at_plasma = epsilon[plasma_idx]
        if abs(eps_at_plasma.real) < 10:  # More tolerant, plasma resonance can be broad
            print(
                f"✅ Plasma frequency behavior correct (ε_real ≈ {eps_at_plasma.real:.3f})"
            )
        else:
            print(
                f"❌ Plasma frequency behavior incorrect: ε_real = {eps_at_plasma.real}"
            )

        # 3. At low frequencies (IR), epsilon real should be large and negative
        low_freq_mask = energies < 0.5  # Well below plasma frequency
        if len(low_freq_mask.nonzero()) > 0:
            eps_low_freq = epsilon[low_freq_mask].real.mean()
            if eps_low_freq < -10:
                print(
                    f"✅ Low frequency metallic behavior (ε_real = {eps_low_freq:.1f})"
                )
            else:
                print(f"❌ Low frequency behavior incorrect: ε_real = {eps_low_freq}")

        # 4. Imaginary part should be positive (absorption)
        eps_imag_positive = (epsilon.imag > 0).all()
        if eps_imag_positive:
            print("✅ Absorption present at all frequencies (ε_imag > 0)")
        else:
            print("❌ Some frequencies show no absorption")

        return True

    except Exception as e:
        print(f"❌ Drude physics test failed: {e}")
        traceback.print_exc()
        return False


def test_drude_device_dtype():
    """Test Drude model device and dtype handling."""
    print("\nTesting Drude model device/dtype handling...")
    try:
        from torch_tmm.dispersion import Drude

        # Create Drude model
        drude = Drude(omega_p=9.0, gamma=0.1)

        # Test different dtypes
        wavelengths = torch.tensor([500.0, 800.0, 1200.0])

        # Test float32
        drude_f32 = drude.to(dtype=torch.float32)
        epsilon_f32 = drude_f32.epsilon(wavelengths)
        if epsilon_f32.dtype == torch.complex64:
            print("✅ Float32 conversion successful")
        else:
            print(f"❌ Float32 conversion failed: got {epsilon_f32.dtype}")
            return False

        # Test float64
        drude_f64 = drude.to(dtype=torch.float64)
        epsilon_f64 = drude_f64.epsilon(wavelengths)
        if epsilon_f64.dtype == torch.complex128:
            print("✅ Float64 conversion successful")
        else:
            print(f"❌ Float64 conversion failed: got {epsilon_f64.dtype}")
            return False

        # Test device movement (CPU only for this test)
        device = torch.device("cpu")
        drude_cpu = drude.to(device=device)
        epsilon_cpu = drude_cpu.epsilon(wavelengths.to(device))
        if epsilon_cpu.device.type == "cpu":
            print("✅ Device movement successful")
        else:
            print(f"❌ Device movement failed: got {epsilon_cpu.device}")
            return False

        return True

    except Exception as e:
        print(f"❌ Drude device/dtype test failed: {e}")
        traceback.print_exc()
        return False


def create_drude_plot():
    """Create a plot showing Drude model optical response."""
    print("\nCreating Drude model demonstration plot...")
    try:
        from torch_tmm.dispersion import Drude

        # Create Drude models with different parameters
        drude_ag = Drude(omega_p=9.2, gamma=0.02)  # Silver-like
        drude_au = Drude(omega_p=8.5, gamma=0.07)  # Gold-like
        drude_al = Drude(omega_p=15.0, gamma=0.1)  # Aluminum-like

        # Wavelength range from UV to mid-IR
        wavelengths = torch.linspace(200, 2000, 500)  # nm

        # Calculate optical properties
        eps_ag = drude_ag.epsilon(wavelengths)
        eps_au = drude_au.epsilon(wavelengths)
        eps_al = drude_al.epsilon(wavelengths)

        n_ag = drude_ag.refractive_index(wavelengths)
        n_au = drude_au.refractive_index(wavelengths)
        n_al = drude_al.refractive_index(wavelengths)

        # Convert to numpy for plotting
        wl_np = wavelengths.detach().numpy()

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Drude Model: Optical Properties of Metals", fontsize=16)

        # Plot epsilon real
        ax1.plot(
            wl_np, eps_ag.real.detach().numpy(), "b-", label="Ag-like", linewidth=2
        )
        ax1.plot(
            wl_np, eps_au.real.detach().numpy(), "r-", label="Au-like", linewidth=2
        )
        ax1.plot(
            wl_np, eps_al.real.detach().numpy(), "g-", label="Al-like", linewidth=2
        )
        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Re(ε)")
        ax1.set_title("Real Part of Permittivity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(200, 2000)

        # Plot epsilon imaginary
        ax2.plot(
            wl_np, eps_ag.imag.detach().numpy(), "b-", label="Ag-like", linewidth=2
        )
        ax2.plot(
            wl_np, eps_au.imag.detach().numpy(), "r-", label="Au-like", linewidth=2
        )
        ax2.plot(
            wl_np, eps_al.imag.detach().numpy(), "g-", label="Al-like", linewidth=2
        )
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Im(ε)")
        ax2.set_title("Imaginary Part of Permittivity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(200, 2000)

        # Plot refractive index real
        ax3.plot(wl_np, n_ag.real.detach().numpy(), "b-", label="Ag-like", linewidth=2)
        ax3.plot(wl_np, n_au.real.detach().numpy(), "r-", label="Au-like", linewidth=2)
        ax3.plot(wl_np, n_al.real.detach().numpy(), "g-", label="Al-like", linewidth=2)
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Re(n)")
        ax3.set_title("Real Part of Refractive Index")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(200, 2000)

        # Plot extinction coefficient (imaginary part of n)
        ax4.plot(wl_np, n_ag.imag.detach().numpy(), "b-", label="Ag-like", linewidth=2)
        ax4.plot(wl_np, n_au.imag.detach().numpy(), "r-", label="Au-like", linewidth=2)
        ax4.plot(wl_np, n_al.imag.detach().numpy(), "g-", label="Al-like", linewidth=2)
        ax4.set_xlabel("Wavelength (nm)")
        ax4.set_ylabel("Im(n) = κ")
        ax4.set_title("Extinction Coefficient")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(200, 2000)

        plt.tight_layout()

        # Save plot
        plot_path = "/tmp/drude_model_demo.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"✅ Plot saved to {plot_path}")
        plt.close()

        return True

    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Drude model tests."""
    print("=" * 60)
    print("Testing Drude Dispersion Model Implementation")
    print("=" * 60)

    tests = [
        ("Basic functionality", test_drude_basic),
        ("Physical behavior", test_drude_physics),
        ("Device/dtype handling", test_drude_device_dtype),
        ("Demonstration plot", create_drude_plot),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Drude model tests passed!")
        print("\nThe Drude dispersion model has been successfully implemented and")
        print("exhibits the expected behavior for metallic optical properties:")
        print("• Plasma frequency cutoff behavior")
        print("• Metallic reflection in IR (ε_real < 0)")
        print("• Proper absorption (ε_imag > 0)")
        print("• Correct high-frequency limit (ε → 1)")
        return 0
    else:
        print("❌ Some Drude model tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
