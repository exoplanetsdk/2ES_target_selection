import matplotlib.pyplot as plt
import seaborn as sns
from config import *
import pandas as pd
import numpy as np

def plot_density_vs_logg(df, logg_col='logg_gaia', density_col='Density [Solar unit]', 
                        density_threshold=0.1, plot_ranges=None, output_path=None, show_plot=False):
    """
    Create two scatter plots of stellar density vs. log g with different zoom levels.
    
    Args:
        df (pd.DataFrame): DataFrame containing the stellar data
        logg_col (str): Name of the log g column
        density_col (str): Name of the density column
        density_threshold (float): Threshold for color-coding points (red below, blue above)
        plot_ranges (dict): Dictionary containing plot ranges. If None, uses defaults.
                          Format: {
                              'full': {'x': (min, max), 'y': (min, max)},
                              'zoom': {'x': (min, max), 'y': (min, max)}
                          }

    Example:
        custom_ranges = {
            'full': {'x': (1.5, 5.5), 'y': (-0.2, 7)},
            'zoom': {'x': (2, 4), 'y': (-0.02, 0.4)}
        }
        plot_density_vs_logg(df, plot_ranges=custom_ranges)                          
    """
    # Ensure log g is numeric
    df[logg_col] = pd.to_numeric(df[logg_col], errors='coerce')
    
    # Default plot ranges if none provided
    default_ranges = {
        'full': {'x': (2, 5), 'y': (-0.1, 6)},
        'zoom': {'x': (2, 4.5), 'y': (-0.01, 0.3)}
    }
    plot_ranges = plot_ranges or default_ranges
    
    # Create color array based on density threshold
    colors = np.where(df[density_col] < density_threshold, 'red', 'blue')
    
    # Common plot parameters
    plot_params = {
        'alpha': 0.6,
        'edgecolors': 'w',
        's': 20,
        'c': colors
    }
    
    # Create both plots
    for plot_type, ranges in plot_ranges.items():
        plt.figure(figsize=(10, 6))
        
        plt.scatter(df[logg_col], df[density_col], **plot_params)
        
        plt.xlabel(logg_col, fontsize=14)
        plt.ylabel(density_col, fontsize=14)
        plt.ylim(*ranges['y'])
        plt.xlim(*ranges['x'])
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_path:
            plt.savefig(output_path[:-4] + f'_{plot_type}.png', dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()

#---------------------------------------------------------------------------------------------------

def plot_color_histogram(df, output_path=None, figsize=(6, 4), dpi=150, show_plot=False):
    """
    Create a histogram of stellar color (BP - RP).
    
    Args:
        df (pd.DataFrame): DataFrame containing the photometric data
        output_path (str, optional): Path to save the plot. If None, plot won't be saved
        figsize (tuple): Figure size in inches (width, height)
        dpi (int): Dots per inch for the figure
        show_plot (bool): Whether to display the plot. Default is True.
        
    Returns:
        None
    """
    # Calculate color
    color = df['Phot BP Mean Mag'] - df['Phot RP Mean Mag']
    
    # Create the plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(color, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Color (BP - RP)')
    plt.ylabel('Frequency')
    plt.title('Color (BP - RP) Histogram')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()

def plot_color_magnitude_diagram(df, output_path=None, figsize=(6, 4), dpi=150, show_plot=False):
    """
    Create a color-magnitude diagram comparing G and V magnitudes.
    
    Args:
        df (pd.DataFrame): DataFrame containing the photometric data
        output_path (str, optional): Path to save the plot. If None, plot won't be saved
        figsize (tuple): Figure size in inches (width, height)
        dpi (int): Dots per inch for the figure
        show_plot (bool): Whether to display the plot. Default is True.
    Returns:
        None
    """
    # Calculate color
    color = df['Phot BP Mean Mag'] - df['Phot RP Mean Mag']
    
    # Calculate conversion factor for colors between 1 and 4
    conv = 0.20220 + 0.02489 * color
    
    # Calculate V magnitude
    V_mag = np.where(
        (color >= 1) & (color <= 4),
        df['Phot BP Mean Mag'] - conv,
        df['Phot G Mean Mag']
    )

    df.insert(df.columns.get_loc('DEC') + 1, 'V_mag', V_mag)
    
    # Create the plot
    plt.figure(figsize=figsize, dpi=dpi)
    
    scatter = plt.scatter(
        df['Phot G Mean Mag'], 
        V_mag, 
        c=color, 
        cmap='viridis', 
        edgecolor='k', 
        s=50, 
        alpha=0.5
    )
    
    plt.xlabel('G Magnitude')
    plt.ylabel('V Magnitude')
    plt.colorbar(scatter, label='Color $(G_{BP} - G_{RP})$')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()

    return df