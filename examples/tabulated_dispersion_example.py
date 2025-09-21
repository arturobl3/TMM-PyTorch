"""
Example usage of the TabulatedData dispersion class.

This script demonstrates how to use the TabulatedData dispersion model
with experimental or computed optical data.
"""

import torch
import sys
import os

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # Define plt as None when not available
    print("Note: matplotlib not available, skipping plots")

# Add the TMM-PyTorch root directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_tmm.dispersion import TabulatedData


def create_sample_data():
    """Create sample refractive index data for demonstration."""
    # Wavelength range from 400 to 1000 nm
    wavelengths = torch.linspace(400, 1000, 31)

    # Create a realistic-looking refractive index
    # Real part: dispersive behavior (higher at shorter wavelengths)
    n_real = 1.5 + 0.1 / (wavelengths / 1000) ** 2

    # Imaginary part: absorption band around 600 nm
    center = 600.0
    width = 50.0
    n_imag = 0.001 + 0.02 * torch.exp(-(((wavelengths - center) / width) ** 2))

    refractive_index = n_real + 1j * n_imag

    return wavelengths, refractive_index


def demonstrate_basic_usage():
    """Demonstrate basic usage of TabulatedData."""
    print("=== Basic TabulatedData Usage ===")

    # Create sample data
    wavelengths_table, n_table = create_sample_data()

    # Create the dispersion model
    tabulated_disp = TabulatedData(wavelengths_table, n_table)
    print(f"Created: {tabulated_disp}")

    # Evaluate at new wavelengths (finer grid)
    eval_wavelengths = torch.linspace(450, 950, 101)
    n_interpolated = tabulated_disp.refractive_index(eval_wavelengths)
    epsilon_interpolated = tabulated_disp.epsilon(eval_wavelengths)

    print(f"Interpolated {len(eval_wavelengths)} wavelength points")
    print(
        f"Refractive index range: {n_interpolated.real.min():.3f} - {n_interpolated.real.max():.3f}"
    )
    print(
        f"Absorption range: {n_interpolated.imag.min():.4f} - {n_interpolated.imag.max():.4f}"
    )

    return (
        wavelengths_table,
        n_table,
        eval_wavelengths,
        n_interpolated,
        epsilon_interpolated,
    )


def demonstrate_bounds_checking():
    """Demonstrate bounds checking behavior - errors for wavelengths outside the tabulated range."""
    print("\n=== Bounds Checking Behavior ===")

    # Create limited data range
    wavelengths_table = torch.linspace(500, 800, 16)
    n_table = 1.5 + 0.01j * torch.ones_like(wavelengths_table)

    tabulated_disp = TabulatedData(wavelengths_table, n_table)

    print(f"Table range: {wavelengths_table[0]:.0f} - {wavelengths_table[-1]:.0f} nm")

    # Test wavelength within range (should work)
    print("\nTesting wavelength within range:")
    within_range_wl = torch.tensor([650.0])
    try:
        n_within = tabulated_disp.refractive_index(within_range_wl)
        print(f"  ✓ λ = {within_range_wl[0]:.0f} nm: n = {n_within[0]:.4f}")
    except ValueError as e:
        print(f"  ✗ Unexpected error: {e}")

    # Test wavelength below range (should raise error)
    print("\nTesting wavelength below range:")
    below_range_wl = torch.tensor([300.0])
    try:
        tabulated_disp.refractive_index(below_range_wl)
        print(f"  ✗ Should have raised error for λ = {below_range_wl[0]:.0f} nm")
    except ValueError as e:
        print(f"  ✓ Expected error: {e}")

    # Test wavelength above range (should raise error)
    print("\nTesting wavelength above range:")
    above_range_wl = torch.tensor([1200.0])
    try:
        tabulated_disp.refractive_index(above_range_wl)
        print(f"  ✗ Should have raised error for λ = {above_range_wl[0]:.0f} nm")
    except ValueError as e:
        print(f"  ✓ Expected error: {e}")

    print("\nBounds checking prevents extrapolation and ensures data integrity!")

    # Return data for plotting (only within bounds)
    eval_wl_limited = torch.linspace(500, 800, 31)  # Within table range
    n_limited = tabulated_disp.refractive_index(eval_wl_limited)

    return wavelengths_table, n_table, eval_wl_limited, n_limited


def plot_results(wavelengths_table, n_table, eval_wavelengths, n_interpolated):
    """Plot the tabulated data and interpolated results."""
    print("\n=== Plotting Results ===")

    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot real part
    ax1.plot(
        wavelengths_table.numpy(),
        n_table.real.numpy(),
        "ro",
        label="Tabulated data",
        markersize=6,
    )
    ax1.plot(
        eval_wavelengths.numpy(),
        n_interpolated.real.numpy(),
        "b-",
        label="Interpolated",
        linewidth=2,
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Real part of n")
    ax1.set_title("Real Part of Refractive Index")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot imaginary part
    ax2.plot(
        wavelengths_table.numpy(),
        n_table.imag.numpy(),
        "ro",
        label="Tabulated data",
        markersize=6,
    )
    ax2.plot(
        eval_wavelengths.numpy(),
        n_interpolated.imag.numpy(),
        "b-",
        label="Interpolated",
        linewidth=2,
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Imaginary part of n")
    ax2.set_title("Imaginary Part of Refractive Index (Absorption)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/tabulated_dispersion_example.png", dpi=150, bbox_inches="tight")
    print("Plot saved to /tmp/tabulated_dispersion_example.png")

    # Don't show the plot in headless environment
    # plt.show()


def demonstrate_from_experimental_data():
    """Demonstrate loading data that might come from experimental measurements."""
    print("\n=== Simulated Experimental Data Usage ===")

    # Simulate loading experimental data (wavelengths in nm, complex n)
    # This could come from ellipsometry, spectroscopy, etc.
    wavelengths_exp = torch.tensor(
        [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0]
    )

    # Simulated experimental refractive indices
    n_exp = torch.tensor(
        [
            1.52 + 0.001j,
            1.51 + 0.002j,
            1.505 + 0.005j,
            1.502 + 0.008j,
            1.500 + 0.015j,
            1.498 + 0.010j,
            1.497 + 0.006j,
            1.496 + 0.004j,
            1.495 + 0.003j,
            1.494 + 0.002j,
            1.493 + 0.001j,
        ]
    )

    print(f"Loaded {len(wavelengths_exp)} experimental data points")

    # Create dispersion model
    exp_dispersion = TabulatedData(wavelengths_exp, n_exp)

    # Use for optical calculations at arbitrary wavelengths
    simulation_wavelengths = torch.linspace(420, 880, 47)  # Different grid
    n_for_simulation = exp_dispersion.refractive_index(simulation_wavelengths)

    print(f"Interpolated to {len(simulation_wavelengths)} simulation points")
    print(
        f"Example value at 650 nm: n = {n_for_simulation[len(n_for_simulation)//2]:.4f}"
    )
    print("Ready for use in TMM optical simulations!")

    return exp_dispersion


if __name__ == "__main__":
    print("TabulatedData Dispersion Class Example")
    print("=" * 50)

    # Run demonstrations
    wl_table, n_table, wl_eval, n_interp, eps_interp = demonstrate_basic_usage()

    demonstrate_bounds_checking()

    # Only plot if matplotlib is available and not in headless mode
    try:
        plot_results(wl_table, n_table, wl_eval, n_interp)
    except Exception as e:
        print(f"\nSkipping plot due to: {e}")

    exp_disp = demonstrate_from_experimental_data()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nThe TabulatedData class is now ready for use in your TMM simulations.")
    print("You can use it to incorporate experimental or literature optical data")
    print("into your thin-film optical models.")
