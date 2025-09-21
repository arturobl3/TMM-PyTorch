"""
Thin Film Optical Simulation Example using TabulatedData

This example demonstrates how to calculate transmission and reflection spectra
for a thin film stack using the TabulatedData dispersion class. The example
simulates a lossy thin film deposited on an intrinsic silicon substrate in air.

The calculation shows:
1. How to set up materials with TabulatedData dispersion
2. How to build a thin film stack with Layer and Model classes
3. How to calculate transmission and reflection for s and p polarizations
4. How to plot and analyze the optical response

Stack structure: Air | Thin Film (variable thickness) | Silicon substrate
"""

import torch
import sys
import os

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

    # Create a dummy plt module to avoid lint errors
    class DummyAxis:
        def plot(*args, **kwargs):
            pass

        def set_xlabel(*args, **kwargs):
            pass

        def set_ylabel(*args, **kwargs):
            pass

        def set_title(*args, **kwargs):
            pass

        def legend(*args, **kwargs):
            pass

        def grid(*args, **kwargs):
            pass

        def set_ylim(*args, **kwargs):
            pass

    class DummyPlt:
        @staticmethod
        def subplots(nrows=1, ncols=1, *args, **kwargs):
            axes = [[DummyAxis() for _ in range(ncols)] for _ in range(nrows)]
            if nrows == 1 and ncols == 1:
                return None, DummyAxis()
            elif nrows == 1:
                return None, tuple(axes[0])
            elif ncols == 1:
                return None, tuple(axes[i][0] for i in range(nrows))
            else:
                return None, tuple(tuple(row) for row in axes)

        @staticmethod
        def tight_layout(*args, **kwargs):
            pass

        @staticmethod
        def savefig(*args, **kwargs):
            pass

    plt = DummyPlt()
    print("Note: matplotlib not available, skipping plots")

# Add the TMM-PyTorch root directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_tmm.dispersion import TabulatedData
from torch_tmm import Material, Layer, Model, Dispersion


def create_sample_thin_film_data():
    """
    Create sample optical data for a lossy thin film material.

    This simulates measured refractive index data that might come from
    ellipsometry or other optical characterization techniques.

    Returns:
        tuple: (wavelengths, complex_refractive_index) tensors
    """
    # Wavelength range: 400-1000 nm (typical for visible-NIR spectroscopy)
    wavelengths = torch.linspace(400, 1000, 61)

    # Create realistic dispersion for a lossy thin film
    # Real part: normal dispersion (higher index at shorter wavelengths)
    n_real = 2.0 + 0.5 / (wavelengths / 1000) ** 2

    # Imaginary part: absorption with a peak around 600 nm
    center = 600.0
    width = 100.0
    n_imag = 0.01 + 0.2 * torch.exp(-(((wavelengths - center) / width) ** 2))

    complex_refractive_index = n_real + 1j * n_imag

    return wavelengths, complex_refractive_index


def calculate_thin_film_optics(film_thickness_nm=100.0, angle_deg=0.0):
    """
    Calculate transmission and reflection for a thin film stack.

    Stack: Air | Thin Film (thickness) | Silicon substrate

    Parameters:
        film_thickness_nm (float): Thickness of the thin film in nanometers
        angle_deg (float): Incident angle in degrees

    Returns:
        dict: Dictionary containing wavelengths, transmission, and reflection data
    """
    print(
        f"Calculating optics for {film_thickness_nm:.1f} nm film at {angle_deg:.1f}° incidence"
    )

    # Create sample thin film optical data
    wavelengths, complex_ri = create_sample_thin_film_data()
    print(f"Created optical data for {len(wavelengths)} wavelength points")

    # Define materials
    print("Setting up materials...")

    # Air environment (n = 1.0, ε = 1.0) - much simpler now!
    air_mat = Material([Dispersion.Constant_epsilon(epsilon_const=1.0)], name="Air")
    air_layer_inf = Layer(air_mat, layer_type="semi-inf")

    # Intrinsic silicon substrate (n ≈ 3.56, ε ≈ 12.7) - much simpler now!
    intrinsic_si_mat = Material(
        [Dispersion.Constant_epsilon(epsilon_const=12.6964494)],
        name="intrinsic-Si",
        requires_grad=False,
    )
    intrinsic_si_layer_inf = Layer(intrinsic_si_mat, layer_type="semi-inf")

    # Lossy thin film with tabulated dispersion data
    thin_film_mat = Material(
        [TabulatedData(wavelengths, complex_ri)],
        name="lossy_thin_film",
        requires_grad=False,
    )

    # Create finite thickness layer for the thin film
    thin_film_layer = Layer(thin_film_mat, thickness=film_thickness_nm)

    print("Materials created:")
    print(f"  - Environment: {air_mat.name}")
    print(f"  - Thin film: {thin_film_mat.name} ({film_thickness_nm:.1f} nm)")
    print(f"  - Substrate: {intrinsic_si_mat.name}")

    # Build the optical model
    optical_model = Model(
        env=air_layer_inf, structure=[thin_film_layer], subs=intrinsic_si_layer_inf
    )

    # Set up calculation parameters
    angles = torch.tensor([angle_deg * torch.pi / 180])  # Convert to radians

    print("Computing optical response...")
    # Compute optical response
    results = optical_model(wavelengths, angles)

    # Extract transmission and reflection for both polarizations
    transmission_s = results.transmission("s").squeeze()  # Remove angle dimension
    transmission_p = results.transmission("p").squeeze()
    reflection_s = results.reflection("s").squeeze()
    reflection_p = results.reflection("p").squeeze()

    # Calculate unpolarized (average) transmission and reflection
    transmission_avg = (transmission_s + transmission_p) / 2.0
    reflection_avg = (reflection_s + reflection_p) / 2.0

    # Calculate absorption (what's not transmitted or reflected)
    absorption_avg = 1.0 - transmission_avg - reflection_avg

    print("Optical calculations completed!")
    print(f"  - Wavelength range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")
    print(f"  - Max transmission: {transmission_avg.max():.3f}")
    print(f"  - Max reflection: {reflection_avg.max():.3f}")
    print(f"  - Max absorption: {absorption_avg.max():.3f}")

    return {
        "wavelengths": wavelengths,
        "complex_ri": complex_ri,
        "transmission_s": transmission_s,
        "transmission_p": transmission_p,
        "transmission_avg": transmission_avg,
        "reflection_s": reflection_s,
        "reflection_p": reflection_p,
        "reflection_avg": reflection_avg,
        "absorption_avg": absorption_avg,
        "film_thickness": film_thickness_nm,
        "angle_deg": angle_deg,
    }


def compare_film_thicknesses():
    """
    Compare optical response for different film thicknesses.

    Returns:
        list: List of results dictionaries for different thicknesses
    """
    print("\\n" + "=" * 60)
    print("THICKNESS COMPARISON STUDY")
    print("=" * 60)

    thicknesses = [50.0, 100.0, 200.0, 400.0]  # nm
    results_list = []

    for thickness in thicknesses:
        print(f"\\nCalculating for {thickness} nm film...")
        result = calculate_thin_film_optics(film_thickness_nm=thickness)
        results_list.append(result)

    return results_list


def plot_optical_response(results):
    """
    Plot transmission, reflection, and absorption spectra.

    Parameters:
        results (dict): Results dictionary from calculate_thin_film_optics
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    wavelengths = results["wavelengths"].numpy()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Refractive index of thin film
    ax1.plot(
        wavelengths,
        results["complex_ri"].real.numpy(),
        "b-",
        label="Real part (n)",
        linewidth=2,
    )
    ax1.plot(
        wavelengths,
        results["complex_ri"].imag.numpy(),
        "r-",
        label="Imaginary part (k)",
        linewidth=2,
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Refractive Index")
    ax1.set_title("Thin Film Optical Constants")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Transmission and Reflection
    ax2.plot(
        wavelengths,
        results["transmission_avg"].numpy(),
        "g-",
        label="Transmission",
        linewidth=2,
    )
    ax2.plot(
        wavelengths,
        results["reflection_avg"].numpy(),
        "b-",
        label="Reflection",
        linewidth=2,
    )
    ax2.plot(
        wavelengths,
        results["absorption_avg"].numpy(),
        "r-",
        label="Absorption",
        linewidth=2,
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Optical Response")
    ax2.set_title(f'Optical Response ({results["film_thickness"]:.0f} nm film)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Plot 3: Polarization comparison (Transmission)
    ax3.plot(
        wavelengths,
        results["transmission_s"].numpy(),
        "b--",
        label="s-polarized",
        linewidth=2,
    )
    ax3.plot(
        wavelengths,
        results["transmission_p"].numpy(),
        "r--",
        label="p-polarized",
        linewidth=2,
    )
    ax3.plot(
        wavelengths,
        results["transmission_avg"].numpy(),
        "k-",
        label="Average",
        linewidth=2,
    )
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Transmission")
    ax3.set_title("Transmission by Polarization")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: Polarization comparison (Reflection)
    ax4.plot(
        wavelengths,
        results["reflection_s"].numpy(),
        "b--",
        label="s-polarized",
        linewidth=2,
    )
    ax4.plot(
        wavelengths,
        results["reflection_p"].numpy(),
        "r--",
        label="p-polarized",
        linewidth=2,
    )
    ax4.plot(
        wavelengths,
        results["reflection_avg"].numpy(),
        "k-",
        label="Average",
        linewidth=2,
    )
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel("Reflection")
    ax4.set_title("Reflection by Polarization")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("/tmp/thin_film_optics_example.png", dpi=150, bbox_inches="tight")
    print("Plot saved to /tmp/thin_film_optics_example.png")


def plot_thickness_comparison(results_list):
    """
    Plot optical response for different film thicknesses.

    Parameters:
        results_list (list): List of results dictionaries
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["blue", "green", "red", "orange"]

    for i, results in enumerate(results_list):
        wavelengths = results["wavelengths"].numpy()
        thickness = results["film_thickness"]
        color = colors[i % len(colors)]

        # Transmission comparison
        ax1.plot(
            wavelengths,
            results["transmission_avg"].numpy(),
            color=color,
            linewidth=2,
            label=f"{thickness:.0f} nm",
        )

        # Reflection comparison
        ax2.plot(
            wavelengths,
            results["reflection_avg"].numpy(),
            color=color,
            linewidth=2,
            label=f"{thickness:.0f} nm",
        )

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmission")
    ax1.set_title("Transmission vs Film Thickness")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflection")
    ax2.set_title("Reflection vs Film Thickness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("/tmp/thickness_comparison.png", dpi=150, bbox_inches="tight")
    print("Thickness comparison plot saved to /tmp/thickness_comparison.png")


def demonstrate_angle_dependence():
    """
    Demonstrate how optical response changes with incident angle.

    Returns:
        list: List of results for different angles
    """
    print("\\n" + "=" * 60)
    print("ANGLE DEPENDENCE STUDY")
    print("=" * 60)

    angles = [0.0, 15.0, 30.0, 45.0, 60.0]  # degrees
    results_list = []

    for angle in angles:
        print(f"\\nCalculating for {angle}° incidence...")
        result = calculate_thin_film_optics(film_thickness_nm=100.0, angle_deg=angle)
        results_list.append(result)

    return results_list


if __name__ == "__main__":
    print("Thin Film Optical Simulation with TabulatedData")
    print("=" * 60)

    # Basic demonstration
    print("\\n1. BASIC THIN FILM CALCULATION")
    print("-" * 40)
    basic_result = calculate_thin_film_optics(film_thickness_nm=100.0)

    # Plot basic results
    try:
        plot_optical_response(basic_result)
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Thickness comparison
    print("\\n2. THICKNESS COMPARISON")
    print("-" * 40)
    thickness_results = compare_film_thicknesses()

    try:
        plot_thickness_comparison(thickness_results)
    except Exception as e:
        print(f"Thickness comparison plotting failed: {e}")

    # Angle dependence
    print("\\n3. ANGLE DEPENDENCE")
    print("-" * 40)
    angle_results = demonstrate_angle_dependence()

    # Summary statistics
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\\nBasic film (100 nm at normal incidence):")
    print(f"  Peak transmission: {basic_result['transmission_avg'].max():.3f}")
    print(f"  Peak reflection: {basic_result['reflection_avg'].max():.3f}")
    print(f"  Peak absorption: {basic_result['absorption_avg'].max():.3f}")

    print("\\nThickness effects:")
    for result in thickness_results:
        t = result["film_thickness"]
        max_t = result["transmission_avg"].max()
        max_r = result["reflection_avg"].max()
        print(f"  {t:3.0f} nm: T_max={max_t:.3f}, R_max={max_r:.3f}")

    print("\\nAngle effects (100 nm film):")
    for result in angle_results:
        angle = result["angle_deg"]
        max_t = result["transmission_avg"].max()
        max_r = result["reflection_avg"].max()
        print(f"  {angle:2.0f}°: T_max={max_t:.3f}, R_max={max_r:.3f}")

    print("\\n" + "=" * 60)
    print("Example completed successfully!")
    print("\\nThis example demonstrates:")
    print("• TabulatedData dispersion for experimental/computed optical data")
    print("• Thin film stack construction with Layer and Model classes")
    print("• Transmission and reflection calculations for s/p polarizations")
    print("• Effects of film thickness and incident angle")
    print("• Integration of TMM-PyTorch components for realistic simulations")
    print("=" * 60)
