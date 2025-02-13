import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def plot_results(wavelengths, angles, R, T):
    """
    Generate a combined graph for all angles and separate graphs for each angle using subplots.
    """
    num_angles = len(angles)
    fig, axes = plt.subplots(num_angles + 1, 1, figsize=(10, 6 * (num_angles + 1)))
    
    # Combined plot for all angles
    for a_idx, angle in enumerate(angles):
        axes[0].plot(wavelengths * 1e9, R[:, a_idx], label=f"Angle {angle}° - R", linestyle='--')
        axes[0].plot(wavelengths * 1e9, T[:, a_idx], label=f"Angle {angle}° - T")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance/Transmittance")
    axes[0].set_title("Optical Response of Thin Films - All Angles")
    axes[0].legend()
    axes[0].grid(True)
    
    # Generate separate subplots for each angle
    for a_idx, angle in enumerate(angles):
        axes[a_idx + 1].plot(wavelengths * 1e9, R[:, a_idx], label=f"Reflectance (R) at {angle}°", linestyle='--', color='blue')
        axes[a_idx + 1].plot(wavelengths * 1e9, T[:, a_idx], label=f"Transmittance (T) at {angle}°", color='green')
        axes[a_idx + 1].set_xlabel("Wavelength (nm)")
        axes[a_idx + 1].set_ylabel("Optical Response")
        axes[a_idx + 1].set_title(f"Optical Response at {angle}°")
        axes[a_idx + 1].legend()
        axes[a_idx + 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def interactive_plot(wavelengths, angles, R, T):
    """
    Create an interactive plot to show both the full graph and a specific angle chosen by the user.
    """
    def update_plot(angle_index):
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths * 1e9, R[:, angle_index], label=f"Reflectance (R) at {angles[angle_index]}°", linestyle='--', color='blue')
        plt.plot(wavelengths * 1e9, T[:, angle_index], label=f"Transmittance (T) at {angles[angle_index]}°", color='green')
        plt.xlabel("Wavelength (nm")
        plt.ylabel("Optical Response")
        plt.title(f"Optical Response at {angles[angle_index]}°")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    angle_slider = widgets.IntSlider(min=0, max=len(angles)-1, step=1, description="Angle Index")
    interactive_plot = widgets.interactive(update_plot, angle_index=angle_slider)
    display(interactive_plot)