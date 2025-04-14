import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate

# Define file paths
buffer_file_paths = ['DataOutput/data_ch1.csv', 'DataOutput/data_ch2.csv']
output_file_paths = ['ModelOutput/output_ch1.csv', 'ModelOutput/output_ch2.csv']

# Track available plots
available_plots = []

# Load buffer data
buffer_data = {}
for i, file_path in enumerate(buffer_file_paths):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        buffer_data[i] = pd.read_csv(file_path, header=None)
        available_plots.append(f"Buffer CH{i+1}")

# Load output data
output_data = {}
for i, file_path in enumerate(output_file_paths):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        output_data[i] = pd.read_csv(file_path, header=None, skiprows=1, dtype=float, skipinitialspace=True)
        available_plots.append(f"Output CH{i+1}")

# Determine the number of plots needed
num_plots = len(buffer_data) + 2 * len(output_data)  # 1 buffer plot per channel, 2 output plots per channel

if num_plots == 0:
    print("No data available to plot.")
    exit()

# Create subplots dynamically
fig, axs = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), squeeze=False)
axs = axs.flatten()  # Ensure axs is always a list

plot_index = 0

# Dictionary to store subplot relationships for zooming
acquired_sample_axes = []
output_axes = []

# Plot buffer data
for i, data in buffer_data.items():
    flattened_data = data.values.flatten()
    indices = np.arange(len(flattened_data))

    axs[plot_index].plot(indices, flattened_data, marker='o', linestyle='-', markersize=2, label=f'Amplitudes CH{i+1}')
    axs[plot_index].set_title(f'Visualization of Acquired Samples CH{i+1}', fontsize=12)
    axs[plot_index].set_xlabel('Sample Index', fontsize=8)
    axs[plot_index].set_ylabel('Amplitude', fontsize=8)
    axs[plot_index].legend(fontsize=8, loc='upper right')
    axs[plot_index].grid(True)

    acquired_sample_axes.append(axs[plot_index])  # Store acquired samples axes for zooming

    plot_index += 1  # Move to the next subplot

# Plot output data
for i, data in output_data.items():
    output_indices = data[0].astype(int)
    displacement_values = data[1]
    time_taken = data[2]

    # Compute latency metrics
    avg_latency = np.mean(time_taken)
    max_latency = np.max(time_taken)
    print(f"Average Latency CH{i+1}: {avg_latency:.2f} ms")
    print(f"Max Latency CH{i+1}: {max_latency:.2f} ms")

    # Integrate displacement (velocity) to get position
    position_values = integrate.cumulative_trapezoid(displacement_values, output_indices, initial=0)

    if plot_index < num_plots:
        axs[plot_index].plot(output_indices, displacement_values, marker='o', linestyle='-', markersize=2, label=f'Displacement Velocity CH{i+1}')
        axs[plot_index].set_title(f'Visualization of Displacement Velocity CH{i+1}', fontsize=12)
        axs[plot_index].set_xlabel('Sample Index', fontsize=8)
        axs[plot_index].set_ylabel('Displacement Velocity', fontsize=8)
        axs[plot_index].legend(fontsize=8, loc='upper right')
        axs[plot_index].grid(True)

        output_axes.append(axs[plot_index])  # Store output-related axes for zooming
        plot_index += 1

    if plot_index < num_plots:
        # Plot points with latency >= 1ms in red
        red_points = time_taken >= 1  # Points with latency >= 1ms
        axs[plot_index].plot(output_indices[red_points], time_taken[red_points], marker='o', linestyle=' ', color='red', markersize=4, label=f'Time Taken > 1ms (CH{i+1})')
        
        # Plot remaining points with the default blue color
        axs[plot_index].plot(output_indices[~red_points], time_taken[~red_points], marker='o', linestyle=' ', markersize=4, label=f'Time Taken < 1ms (CH{i+1})', color='C0')  # 'C0' is the default color

        axs[plot_index].set_title(f'Model Computation Time CH{i+1}', fontsize=12)
        axs[plot_index].set_xlabel('Sample Index', fontsize=8)
        axs[plot_index].set_ylabel('Time (ms)', fontsize=8)
        axs[plot_index].legend(fontsize=8, loc='upper right')
        axs[plot_index].grid(True)

        output_axes.append(axs[plot_index])  # Store output-related axes for zooming
        plot_index += 1



# Adjust layout based on number of plots
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.7, top=0.95, bottom=0.05)

# Synchronization logic for zooming
setting_limits = {'value': False}

def on_xlims_change(event_ax):
    """ Synchronizes zooming across subplots """
    if setting_limits['value']:
        return

    setting_limits['value'] = True
    xlim = event_ax.get_xlim()

    if event_ax in acquired_sample_axes:
        # Acquired samples should be zoomed out 48x relative to output graphs
        for ax in acquired_sample_axes:
            ax.set_xlim(xlim)

        output_xlim = (xlim[0] / 48, xlim[1] / 48)
        for ax in output_axes:
            ax.set_xlim(output_xlim)

    elif event_ax in output_axes:
        # Keep displacement velocity & Model computation time on the same scale
        for ax in output_axes:
            ax.set_xlim(xlim)

        acquired_xlim = (xlim[0] * 48, xlim[1] * 48)
        for ax in acquired_sample_axes:
            ax.set_xlim(acquired_xlim)

    setting_limits['value'] = False
    fig.canvas.draw_idle()

def connect_callbacks():
    """ Connects zoom callbacks to subplots """
    for ax in axs:
        ax.callbacks.connect('xlim_changed', on_xlims_change)

connect_callbacks()

plt.show()
