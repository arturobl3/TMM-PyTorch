import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

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
    out = widgets.Output()
    angle_slider = widgets.IntSlider(min=0, max=len(angles)-1, step=1, description="Angle Index")

    def update_plot(angle_index):
        with out:
            clear_output(wait=True)  # Clear previous output before drawing a new plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(wavelengths * 1e9, R[:, angle_index],
                    label=f"Reflectance (R) at {angles[angle_index]}°",
                    linestyle='--', color='blue')
            ax.plot(wavelengths * 1e9, T[:, angle_index],
                    label=f"Transmittance (T) at {angles[angle_index]}°",
                    color='green')
            ax.plot(wavelengths * 1e9, T[:, angle_index] + R[:, angle_index] ,
                    label=f"T+R at {angles[angle_index]}°",
                    color='red')
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Optical Response")
            ax.set_title(f"Optical Response at {angles[angle_index]}°")
            ax.legend()
            ax.grid(True)
            plt.show()
            

    # Display the slider and the output widget
    display(widgets.VBox([angle_slider, out]))

    # Attach the update function to slider changes
    def on_value_change(change):
        update_plot(change['new'])

    angle_slider.observe(on_value_change, names='value')
    # Draw initial plot
    update_plot(angle_slider.value)

def interactive_plot_sp(wavelengths, angles, Rs, Ts, Rp, Tp):
    out = widgets.Output()
    angle_slider = widgets.IntSlider(min=0, max=len(angles)-1, step=1, description="Angle Index")

    def update_plot(angle_index):
        with out:
            clear_output(wait=True)  # Clear previous output before drawing a new plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(wavelengths * 1e9, Rs[:, angle_index],
                    label=f"Reflectance (R) s-pol",
                    linestyle='--', color='blue')
            ax.plot(wavelengths * 1e9, Ts[:, angle_index],
                    label=f"Transmittance (T) s-pol",
                    color='green')
            ax.plot(wavelengths * 1e9, Rp[:, angle_index] ,
                    label=f"Reflectance (R) p-pol",
                    linestyle='--', color='orange')
            ax.plot(wavelengths * 1e9, Tp[:, angle_index] ,
                    label=f"Transmittance (T) p-pol",
                    color='red')
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Optical Response")
            ax.set_title(f"Optical Response at {angles[angle_index]}°")
            ax.legend()
            ax.grid(True)
            plt.show()
            

    # Display the slider and the output widget
    display(widgets.VBox([angle_slider, out]))

    # Attach the update function to slider changes
    def on_value_change(change):
        update_plot(change['new'])

    angle_slider.observe(on_value_change, names='value')
    # Draw initial plot
    update_plot(angle_slider.value)


    # Display the slider and the output widget
    display(widgets.VBox([angle_slider, out]))

    # Attach the update function to slider changes
    def on_value_change(change):
        update_plot(change['new'])

    angle_slider.observe(on_value_change, names='value')
    # Draw initial plot
    update_plot(angle_slider.value)

def TIR_plot(angles, Rs, Rp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles, Rs[0, :],
            label=f"Reflectance (R) s-pol",
            linestyle='--', color='blue')
    ax.plot(angles, Rp[0, :] ,
            label=f"Reflectance (R) p-pol",
            linestyle='--', color='orange')
    ax.set_xlabel("Angles of incidence (°)")
    ax.set_ylabel("Optical Response")
    ax.set_title(f"Total Internal reflection")
    ax.legend()
    ax.grid(True)
    plt.show()



def interactive_plot_wl_sp(wavelengths, angles, Rs, Ts, Rp, Tp):
    """
    Creates an interactive plot with a slider along wavelengths while the x-axis represents 
    incident angles. Additionally, checkboxes allow toggling of the Rs, Ts, Rp, and Tp curves.

    Parameters
    ----------
    wavelengths : array-like
        1D array (or torch.Tensor) of wavelengths (in meters).
    angles : array-like
        1D array (or torch.Tensor) of incident angles (in degrees).
    Rs, Ts, Rp, Tp : array-like
        2D arrays (or torch.Tensors) with shape 
        (n_wavelengths, n_angles) corresponding to the optical responses.
    """
    # Create an output widget for the plot.
    out = widgets.Output()
    
    # Create a slider for selecting the wavelength index.
    wl_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(wavelengths) - 1,
        step=1,
        description="Wavelength Index",
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    
    # Create checkboxes to toggle curves.
    cb_Rs = widgets.Checkbox(value=True, description="Show Rs")
    cb_Ts = widgets.Checkbox(value=True, description="Show Ts")
    cb_Rp = widgets.Checkbox(value=True, description="Show Rp")
    cb_Tp = widgets.Checkbox(value=True, description="Show Tp")
    
    # Arrange checkboxes in a horizontal box.
    toggle_box = widgets.HBox([cb_Rs, cb_Ts, cb_Rp, cb_Tp])
    
    def update_plot(wl_index):
        with out:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot each response if its corresponding checkbox is checked.
            if cb_Rs.value:
                ax.plot(angles, Rs[wl_index, :],
                        label="Reflectance (R) s-pol",
                        linestyle='--', color='blue')
            if cb_Ts.value:
                ax.plot(angles, Ts[wl_index, :],
                        label="Transmittance (T) s-pol",
                        color='green')
            if cb_Rp.value:
                ax.plot(angles, Rp[wl_index, :],
                        label="Reflectance (R) p-pol",
                        linestyle='--', color='orange')
            if cb_Tp.value:
                ax.plot(angles, Tp[wl_index, :],
                        label="Transmittance (T) p-pol",
                        color='red')
            
            ax.set_xlabel("Incident Angle (°)")
            ax.set_ylabel("Optical Response")
            # Convert wavelength to nm for display.
            current_wl_nm = wavelengths[wl_index] * 1e9  
            ax.set_title(f"Optical Response at {current_wl_nm:.2f} nm (Index {wl_index})")
            ax.legend()
            ax.grid(True)
            plt.show()
    
    # Whenever any widget changes, update the plot.
    wl_slider.observe(lambda change: update_plot(change.new), names='value')
    cb_Rs.observe(lambda change: update_plot(wl_slider.value), names='value')
    cb_Ts.observe(lambda change: update_plot(wl_slider.value), names='value')
    cb_Rp.observe(lambda change: update_plot(wl_slider.value), names='value')
    cb_Tp.observe(lambda change: update_plot(wl_slider.value), names='value')
    
    # Display the slider, checkboxes, and output widget.
    display(widgets.VBox([wl_slider, toggle_box, out]))
    update_plot(wl_slider.value)
