#!/usr/bin/env python3
"""
Example: Using Drude dispersion model for metallic thin films

This example demonstrates how to use the new Drude dispersion model
to simulate the optical properties of metallic thin films on substrates.
"""

import torch
import matplotlib.pyplot as plt
from torch_tmm.dispersion import Drude, Constant_epsilon
from torch_tmm.material import Material
from torch_tmm.layer import Layer
from torch_tmm.model import Model


def metal_on_substrate_example():
    """
    Example: Gold thin film on glass substrate

    This demonstrates a realistic optical simulation using:
    - Drude model for gold metal layer
    - Constant epsilon for air environment
    - Constant epsilon for glass substrate
    """
    print("Metal-on-Substrate Optical Simulation")
    print("=" * 50)

    # Define wavelength range (visible + near-IR)
    wavelengths = torch.linspace(400, 1200, 200)  # nm

    # === MATERIALS ===

    # Environment: Air (n = 1.0)
    air_dispersion = Constant_epsilon(epsilon_const=1.0)
    air = Material(dispersion=[air_dispersion], name="Air")

    # Metal layer: Gold (using Drude model)
    # Parameters roughly based on experimental gold data
    gold_drude = Drude(
        omega_p=8.5, gamma=0.075  # eV - plasma frequency  # eV - collision frequency
    )
    gold = Material(dispersion=[gold_drude], name="Gold")

    # Substrate: Glass (n ≈ 1.5)
    glass_dispersion = Constant_epsilon(epsilon_const=1.5**2)
    glass = Material(dispersion=[glass_dispersion], name="Glass")

    # === LAYER STRUCTURE ===

    # Create layers
    env_layer = Layer(air, layer_type="semi-inf")
    gold_layer = Layer(gold, layer_type="coh", thickness=50.0)  # 50 nm gold
    substrate_layer = Layer(glass, layer_type="semi-inf")

    # Create optical model
    model = Model(env=env_layer, structure=[gold_layer], subs=substrate_layer)

    print(f"Structure: Air | {gold_layer.thickness} nm Gold | Glass")

    # === OPTICAL CALCULATIONS ===

    # Calculate at normal incidence
    angles = torch.tensor([0.0])  # degrees

    try:
        results = model(wavelengths, angles)

        # Extract transmission and reflection for both polarizations
        transmission_s = results.transmission("s").squeeze()  # Remove angle dimension
        transmission_p = results.transmission("p").squeeze()
        reflection_s = results.reflection("s").squeeze()
        reflection_p = results.reflection("p").squeeze()

        # Calculate unpolarized (average) transmission and reflection
        T = (transmission_s + transmission_p) / 2.0
        R = (reflection_s + reflection_p) / 2.0
        A = 1.0 - T - R  # Absorption

        print(f"\nOptical response calculated for {len(wavelengths)} wavelengths")
        print(f"Peak transmission: {T.max():.3f}")
        print(f"Peak reflection: {R.max():.3f}")
        print(f"Peak absorption: {A.max():.3f}")

        # === PLOTTING ===

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot optical response
        wl_nm = wavelengths.detach().numpy()
        ax1.plot(wl_nm, T.detach().numpy(), "b-", label="Transmission", linewidth=2)
        ax1.plot(wl_nm, R.detach().numpy(), "r-", label="Reflection", linewidth=2)
        ax1.plot(wl_nm, A.detach().numpy(), "g-", label="Absorption", linewidth=2)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Optical Response")
        ax1.set_title("50 nm Gold Film on Glass")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(400, 1200)
        ax1.set_ylim(0, 1)

        # Plot complex refractive index of gold
        n_gold = gold_drude.refractive_index(wavelengths)
        ax2.plot(
            wl_nm, n_gold.real.detach().numpy(), "b-", label="n (real)", linewidth=2
        )
        ax2.plot(
            wl_nm,
            n_gold.imag.detach().numpy(),
            "r-",
            label="κ (extinction)",
            linewidth=2,
        )
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Refractive Index")
        ax2.set_title("Gold Optical Constants (Drude Model)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(400, 1200)

        plt.tight_layout()

        # Save plot
        plot_path = "/tmp/drude_metal_example.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

        return True

    except Exception as e:
        print(f"Error in optical calculation: {e}")
        import traceback

        traceback.print_exc()
        return False


def drude_vs_constant_comparison():
    """
    Compare Drude model vs simplified constant model for metals
    """
    print("\n" + "=" * 50)
    print("Drude vs Constant Model Comparison")
    print("=" * 50)

    wavelengths = torch.linspace(500, 1500, 100)

    # Drude model for silver
    silver_drude = Drude(omega_p=9.2, gamma=0.02)

    # Simplified constant model (rough approximation at 800nm)
    # For silver at 800nm: n ≈ 0.05 + 5.0i (very rough)
    silver_n_approx = 0.05 + 5.0j
    silver_constant = Constant_epsilon(epsilon_const=torch.tensor(silver_n_approx**2))

    # Calculate optical properties
    eps_drude = silver_drude.epsilon(wavelengths)
    eps_constant = silver_constant.epsilon(wavelengths)

    n_drude = silver_drude.refractive_index(wavelengths)
    n_constant = silver_constant.refractive_index(wavelengths)

    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Drude vs Constant Model: Silver Optical Properties", fontsize=14)

    wl_nm = wavelengths.detach().numpy()

    # Epsilon real
    ax1.plot(wl_nm, eps_drude.real.detach().numpy(), "b-", label="Drude", linewidth=2)
    ax1.plot(
        wl_nm, eps_constant.real.detach().numpy(), "r--", label="Constant", linewidth=2
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Re(ε)")
    ax1.set_title("Real Permittivity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon imaginary
    ax2.plot(wl_nm, eps_drude.imag.detach().numpy(), "b-", label="Drude", linewidth=2)
    ax2.plot(
        wl_nm, eps_constant.imag.detach().numpy(), "r--", label="Constant", linewidth=2
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Im(ε)")
    ax2.set_title("Imaginary Permittivity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # n real
    ax3.plot(wl_nm, n_drude.real.detach().numpy(), "b-", label="Drude", linewidth=2)
    ax3.plot(
        wl_nm, n_constant.real.detach().numpy(), "r--", label="Constant", linewidth=2
    )
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Re(n)")
    ax3.set_title("Refractive Index (Real)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # n imaginary (extinction)
    ax4.plot(wl_nm, n_drude.imag.detach().numpy(), "b-", label="Drude", linewidth=2)
    ax4.plot(
        wl_nm, n_constant.imag.detach().numpy(), "r--", label="Constant", linewidth=2
    )
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel("Im(n) = κ")
    ax4.set_title("Extinction Coefficient")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save comparison plot
    plot_path = "/tmp/drude_vs_constant_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()

    print("\nKey differences:")
    print("• Drude model shows wavelength-dependent dispersion")
    print("• Constant model gives same values at all wavelengths")
    print("• Drude model captures physical plasma frequency effects")
    print("• For accurate simulations, use Drude or tabulated data")


def main():
    """Run Drude model examples"""
    print("Drude Dispersion Model Examples")
    print("=" * 60)

    success1 = metal_on_substrate_example()
    drude_vs_constant_comparison()

    print("\n" + "=" * 60)
    if success1:
        print("✅ Drude model examples completed successfully!")
        print("\nThe Drude model provides:")
        print("• Physically accurate metallic dispersion")
        print("• Wavelength-dependent optical properties")
        print("• Proper plasma frequency behavior")
        print("• Integration with TMM optical simulations")
    else:
        print("❌ Some examples failed")

    print("\nFor realistic metal simulations, consider:")
    print("• Using tabulated experimental data when available")
    print("• Combining Drude + interband transitions for noble metals")
    print("• Adjusting plasma/collision frequencies for specific metals")


if __name__ == "__main__":
    main()
