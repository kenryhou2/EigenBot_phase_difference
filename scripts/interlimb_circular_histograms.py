#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from scipy.stats import circmean, circstd


# def plot_circular_histogram(ax, datasets, colors, labels, variable_name):
#     """
#     Plot a circular histogram with mean and standard deviation on the provided axes.

#     Parameters:
#     ax (matplotlib.axes._subplots.PolarAxesSubplot): The polar subplot axes to plot on.
#     datasets (list of arrays): List of arrays containing phase differences in radians.
#     colors (list of str): List of colors for each dataset.
#     labels (list of str): List of labels for each dataset.
#     variable_name (str): Name of the variable (used as the title of the plot).
#     """
#     for data, color, label in zip(datasets, colors, labels):
#         if data is not None and len(data) > 0:
#             mean = circmean(data, high=np.pi, low=-np.pi)
#             std_dev = circstd(data, high=np.pi, low=-np.pi)
            
#             ax.plot([mean, mean], [0, max(ax.get_ylim())], color=color, linewidth=2, label=f'{label} Mean {mean:.3f} rad')
#             ax.fill_betweenx([0, max(ax.get_ylim())], mean - std_dev, mean + std_dev, color=color, alpha=0.3, label=f'{label} SD {std_dev:.3f} rad')
    
#     # Set title with bold font
#     ax.set_title(variable_name, fontweight='bold')

#     # Set polar plot labels to radians (π, π/2, etc.)
#     ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
#     ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'], fontsize=15)
#     ax.set_rticks([0.5, 1])

def plot_circular_histogram(ax, datasets, colors, labels, variable_name):
    """
    Plot a circular histogram with mean and standard deviation on the provided axes.

    Parameters:
    ax (matplotlib.axes._subplots.PolarAxesSubplot): The polar subplot axes to plot on.
    datasets (list of arrays): List of arrays containing phase differences in radians.
    colors (list of str): List of colors for each dataset.
    labels (list of str): List of labels for each dataset.
    variable_name (str): Name of the variable (used as the title of the plot).
    """
    for data, color, label in zip(datasets, colors, labels):
        if data is not None and len(data) > 0:
            # mean = circmean(data, high=np.pi, low=-np.pi)
            # std_dev = circstd(data, high=np.pi, low=-np.pi)
            mean = circmean(data, high=np.pi, low=0)
            std_dev = circstd(data, high=np.pi, low=0)
            
            # Plot the mean line
            ax.plot([mean, mean], [0, max(ax.get_ylim())], color=color, linewidth=2, label=f'{label} Mean {mean:.2f} rad')

            # ax.hist(data, bins=30, color=color, alpha=0.5, label=f'{label} SD {std_dev:.2f} rad')
            
            # Plot the standard deviation as a filled sector along the length of the mean
            theta = np.linspace(mean - std_dev, mean + std_dev, 100)
            r = max(ax.get_ylim()) * np.ones_like(theta)  # Set radius to the edge of the plot
            
            # Fill the sector between the standard deviation bounds
            ax.fill_between(theta, 0, r, color=color, alpha=0.25)
    
    # Set title with bold font
    # ax.set_title(variable_name, fontweight='bold')
    # ax.legend(loc='upper right')

    # Set polar plot labels to radians (π, π/2, etc.)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'], fontsize=20)
    ax.set_rticks([0.5, 1])

def main(file_paths):
    """
    Main function to read data from multiple CSV files and plot circular histograms for each variable.

    Parameters:
    file_paths (list of str): List of paths to the CSV files containing the phase data.
    """
    # Read all CSV files
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]
    
    # Extract file names for labeling
    file_labels = [file_path.split('/')[-1].replace('.csv', '') for file_path in file_paths]
    
    # Define subplot positions to match the layout in the provided image
    positions = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (0, 2), (1, 2)]
    fig, axs = plt.subplots(3, 3, subplot_kw={'polar': True}, figsize=(12, 12))
    
    # Flatten the subplot axes for easy indexing
    axs = axs.flatten()
    
    # Iterate over each column and plot the circular histogram in the specified positions
    for i, (column, pos) in enumerate(zip(dataframes[0].columns, positions)):
        datasets = []
        for df in dataframes:
            if column in df.columns:
                phase_data = np.pi * pd.to_numeric(df[column], errors='coerce').dropna().astype(float).to_numpy()
                datasets.append(phase_data)
            else:
                datasets.append(None)

        # Assign colors to each dataset and use filenames as labels
        colors = [
            'red' 
            ,'blue'
            , 'orange'
            , 'green'
            , 'brown'
            , 'purple'
            ][:len(datasets)]
        labels = file_labels[:len(datasets)]

        # Plot the circular histogram for the current variable
        ax = axs[pos[0] * 3 + pos[1]]  # Access subplot by position
        plot_circular_histogram(ax, datasets, colors, labels, column)
    
    # Hide unused subplots
    for ax in axs:
        if not ax.has_data():
            ax.set_visible(False)

    # Create a combined legend outside the plot
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(dataframes), fontsize='large')

    plt.tight_layout()
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    file_paths = [
        '../data/No_Amputation_Flat.csv'
        ,'../data/Amputate_L2R2.csv'
        ,'../data/Amputate_R3L3.csv'
        ,'../data/Amputate_L3.csv'
        ,'../data/Amputate_R2.csv'
        ,'../data/Ablation.csv'
        # Add more file paths as needed
    ]
    main(file_paths)