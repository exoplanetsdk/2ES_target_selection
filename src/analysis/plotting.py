import matplotlib.pyplot as plt
import seaborn as sns
from config import *
import pandas as pd
import numpy as np
from core.utils import adjust_column_widths

def plot_density_vs_logg(df, logg_col='logg_gaia', density_col='Density [Solar unit]', 
                        density_threshold=0.1, plot_ranges=None, output_path=None, show_plot=False):
    print('\nPlotting density vs logg')
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
            print(f"Saved {output_path[:-4]}_{plot_type}.png")
        
        if show_plot:
            plt.show()

#---------------------------------------------------------------------------------------------------

def plot_color_histogram(df, output_path=None, figsize=(6, 4), dpi=150, show_plot=False):
    print('\nPlotting color histogram')
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
        print(f'Saved {output_path}')
    
    if show_plot:
        plt.show()

#---------------------------------------------------------------------------------------------------

def plot_color_magnitude_diagram(df, output_path=None, figsize=(6, 4), dpi=150, show_plot=False):
    print('\nPlotting color magnitude diagram')
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
        print(f'Saved {output_path}')
    
    if show_plot:
        plt.show()

    return df

#---------------------------------------------------------------------------------------------------

def plot_scatter(x, y, data, xlabel, ylabel, xlim=None, ylim=None, filename=None, color=None, alpha=0.7, 
                 size=60, invert_xaxis=False, x2=None, y2=None, data2=None, color2=None, alpha2=0.7, 
                 size2=60, show_plot=False):
    """Creates and saves a scatter plot with the given parameters, with an option to add a second group of data."""
    plt.figure(figsize=(6, 4), dpi=150)
    
    # Plot the first group of data
    if color is not None:
        sns.scatterplot(x=x, y=y, data=data, color=color, alpha=alpha, s=size)
    else:
        sns.scatterplot(x=x, y=y, data=data, alpha=alpha, s=size)

    # Plot the second group of data if provided
    if x2 is not None and y2 is not None and data2 is not None:
        if color2 is not None:
            sns.scatterplot(x=x2, y=y2, data=data2, color=color2, alpha=alpha2, s=size2, marker='+', linewidth=2)
        else:
            sns.scatterplot(x=x2, y=y2, data=data2, alpha=alpha2, s=size2, marker='+', linewidth=2)

    # Customize the plot
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Invert x-axis if specified
    if invert_xaxis:
        plt.gca().invert_xaxis()

    # Customize tick labels
    plt.tick_params(axis='both', which='major')

    # Adjust layout and display the plot
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        print(f'Saved {filename}')
    if show_plot:
        plt.show()

#---------------------------------------------------------------------------------------------------

def plot_stellar_properties_vs_temperature(df, detection_limit, show_plot=False):
    print('\nPlotting stellar properties vs temperature')
    """
    Plots various stellar properties as a function of effective temperature.

    Parameters:
    merged_df (DataFrame): DataFrame containing stellar data with columns for effective temperature and other properties.
    """
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), dpi=300)

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # List of columns to plot, including the new Density column
    columns = [
        'V_mag', 'Mass [M_Sun]', 'Luminosity [L_Sun]', 'Radius [R_Sun]', 
        'Density [Solar unit]', 'HZ_limit [AU]', 'σ_photon [m/s]', 'HZ Detection Limit [M_Earth]'
    ]

    # Define a color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    # Plot each column
    for i, col in enumerate(columns):
        axs[i].scatter(df['T_eff [K]'], df[col], alpha=0.8, color=colors[i], s=10)
        axs[i].set_xlabel('$T_{eff}$ (K)', fontsize=12)
        axs[i].set_ylabel(col, fontsize=12)
        axs[i].grid(True, linestyle='--', alpha=0.6)

    # Add a main title
    fig.suptitle('Stellar Properties as a Function of Temperature\n(' + str(len(df)) + ' stars with HZ Detection Limit < ' + str(detection_limit) + ' M_Earth)', fontsize=14)

    # Adjust the layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{FIGURES_DIRECTORY}stellar_properties_vs_temperature_{detection_limit}.png')
    print(f'Saved {FIGURES_DIRECTORY}stellar_properties_vs_temperature_{detection_limit}.png')
    plt.show() if show_plot else plt.close()

#---------------------------------------------------------------------------------------------------

def plot_hr_diagram_with_detection_limit(df, use_filtered_data=True, detection_limit=1.5, dpi=150, show_plot=False):
    print('\nPlotting HR diagram with detection limit')
    """
    Create a Hertzsprung-Russell diagram with stars color-coded by their habitable zone detection limit.
    
    Args:
        df (pd.DataFrame): DataFrame containing stellar data
        use_filtered_data (bool, optional): Whether to filter data based on detection limit. Defaults to True
        detection_limit (float, optional): Upper limit for HZ detection limit filtering. Defaults to 1.5
        dpi (int, optional): DPI for the output figure. Defaults to 150
        
    Returns:
        None
    """
    # Determine which data to plot based on filtering option
    if use_filtered_data:
        data_to_plot = df[df['HZ Detection Limit [M_Earth]'] <= detection_limit]
        color_data = data_to_plot['HZ Detection Limit [M_Earth]']
        colorbar_label = f'HZ Detection Limit [M_Earth]'
        filename = f'{FIGURES_DIRECTORY}HR_diagram_HZ_detection_limit_filtered_{detection_limit}.png'
        print(f'Number of stars with HZ Detection Limit <= {detection_limit}: {len(data_to_plot)}')
        plot_stellar_properties_vs_temperature(data_to_plot, detection_limit, show_plot=show_plot)
    else:
        data_to_plot = df
        color_data = np.minimum(df['HZ Detection Limit [M_Earth]'], 4)
        colorbar_label = 'HZ Detection Limit [M_Earth]'
        filename = f'{FIGURES_DIRECTORY}HR_diagram_HZ_detection_limit.png'

    # Create the plot
    plt.figure(figsize=(10, 8), dpi=dpi)

    # Scatter plot with color-coded circles
    sc = plt.scatter(
        data_to_plot['T_eff [K]'], 
        data_to_plot['Luminosity [L_Sun]'], 
        c=color_data,
        cmap='viridis',
        alpha=0.99, 
        edgecolors='grey',
        linewidths=0.05,
        s=data_to_plot['Radius [R_Sun]'] * 20
    )

    # Add and configure colorbar
    cbar = plt.colorbar(sc, label=colorbar_label)
    sc.set_clim(0, 4)

    # Configure axes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(df['T_eff [K]'])-50, max(df['T_eff [K]'])+50)
    plt.ylim(min(df['Luminosity [L_Sun]']), max(df['Luminosity [L_Sun]'])+0.5)
    plt.gca().invert_xaxis()

    # Add labels and title
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Luminosity (L/L_sun)')
    if use_filtered_data:
        plt.title('Hertzsprung-Russell Diagram (' + str(len(data_to_plot)) + ' < ' + str(detection_limit) + ' M_Earth)')
    else:
        plt.title('Hertzsprung-Russell Diagram')
    
    # Add grid
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Save and display the plot
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.show() if show_plot else plt.close()

#---------------------------------------------------------------------------------------------------

def plot_hr_diagram_multi_detection_limits(df, detection_limits=[None, 4, 2, 1.5], dpi=150, show_plot=False):
    """
    Create a Hertzsprung-Russell diagram with multiple subplots for different detection limits,
    all sharing the same colorbar.
    
    Args:
        df (pd.DataFrame): DataFrame containing stellar data
        detection_limits (list): List of detection limits to plot (None for all data)
        dpi (int, optional): DPI for the output figure. Defaults to 150
        show_plot (bool, optional): Whether to show the plot. Defaults to False
        
    Returns:
        None
    """
    print('\nPlotting HR diagram with multiple detection limits')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=dpi, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Get global min/max for temperature and luminosity for consistent axes
    temp_min = min(df['T_eff [K]']) - 50
    temp_max = max(df['T_eff [K]']) + 50
    lum_min = min(df['Luminosity [L_Sun]'])
    lum_max = max(df['Luminosity [L_Sun]']) + 0.5
    
    # Create a list to store scatter plot objects for colorbar
    scatter_plots = []
    
    # Loop through each subplot and detection limit
    for i, detection_limit in enumerate(detection_limits):
        ax = axes[i]
        
        # Determine which data to plot based on detection limit
        if detection_limit is not None:
            data_to_plot = df[df['HZ Detection Limit [M_Earth]'] <= detection_limit]
            subtitle = f'({len(data_to_plot)} < {detection_limit} M_Earth)'
        else:
            data_to_plot = df
            subtitle = f'(All {len(data_to_plot)} stars)'
        
        # Use consistent color mapping across all plots (0-4 M_Earth)
        color_data = np.minimum(data_to_plot['HZ Detection Limit [M_Earth]'], 4)
        
        # Create scatter plot
        sc = ax.scatter(
            data_to_plot['T_eff [K]'], 
            data_to_plot['Luminosity [L_Sun]'], 
            c=color_data,
            cmap='viridis',
            alpha=0.99, 
            edgecolors='grey',
            linewidths=0.05,
            s=data_to_plot['Radius [R_Sun]'] * 20,
            vmin=0,  # Fixed color scale from 0 to 4
            vmax=4
        )
        scatter_plots.append(sc)
        
        # Configure axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(temp_max, temp_min)  # Inverted x-axis
        ax.set_ylim(lum_min, lum_max)
        
        # Add subtitle
        ax.set_title(subtitle)
        
        # Add grid
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        
        # Add labels for outer plots only
        if i >= 2:  # Bottom row
            ax.set_xlabel('Effective Temperature (K)')
        if i % 2 == 0:  # Left column
            ax.set_ylabel('Luminosity (L/L_sun)')
    
    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax)
    cbar.set_label('HZ Detection Limit [M_Earth]')
    
    # Add main title
    fig.suptitle('Hertzsprung-Russell Diagram', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save and display the plot
    filename = f'{FIGURES_DIRECTORY}HR_diagram_multi_detection_limits.png'
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.show() if show_plot else plt.close()

#---------------------------------------------------------------------------------------------------

def analyze_stellar_data(df, hz_limits=None, date_str=None, show_plot=False):
    print(f'\nAnalyzing stellar data and creating histograms for different HZ Detection Limits: {hz_limits}')
    """
    Analyze stellar data and create histograms for different HZ Detection Limits.
    Also saves filtered DataFrames to Excel files.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing stellar data
        hz_limits (list, optional): List of HZ Detection Limits to analyze. Defaults to [None, 4, 1.5]
        date_str (str, optional): Date string for file naming. Defaults to current date
        
    Returns:
        dict: Dictionary containing filtered DataFrames for each HZ limit
    """

    from datetime import datetime
    
    # Set default values
    if hz_limits is None:
        hz_limits = [None, 4, 1.5]
    if date_str is None:
        date_str = datetime.now().strftime('%Y.%m.%d')

    # Set the theme for the plot
    sns.set_theme(style="whitegrid")

    # Calculate distance if not already present
    if 'Distance [pc]' not in df.columns:
        df['Distance [pc]'] = 1000 / df['Parallax']

    # Define columns to plot
    columns_to_plot = [
        'V_mag', 'Phot G Mean Mag', 'Phot BP Mean Mag', 'Distance [pc]',
        'T_eff [K]', 'Luminosity [L_Sun]', 'Mass [M_Sun]', 'Radius [R_Sun]', 
        'Density [Solar unit]', 'HZ_limit [AU]', 'σ_photon [m/s]', 'HZ Detection Limit [M_Earth]'
    ]

    # Define group colors
    group_colors = {
        'Brightest': 'red',
        'Bright': 'orange',
        'Dim': 'green',
        'Dimmer': 'blue',
        'Dimmest': 'black'
    }

    def plot_histograms(data, title, filename, show_plot=show_plot):
        # Divide V_mag into 5 groups
        v_mag_bins = np.linspace(data['V_mag'].min(), data['V_mag'].max(), 6)
        data['V_mag_group'] = pd.cut(data['V_mag'], bins=v_mag_bins, labels=group_colors.keys())

        # Create plot
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=150)
        fig.suptitle(f'{title} (sample = {len(data)})', fontsize=16)

        # Flatten axes array
        axes = axes.flatten()

        # Create histograms
        for i, column in enumerate(columns_to_plot):
            if column in data.columns:
                sns.histplot(
                    data=data, 
                    x=column, 
                    ax=axes[i], 
                    hue='V_mag_group', 
                    palette=group_colors, 
                    multiple='stack', 
                    legend=False
                )
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Count')
            else:
                axes[i].set_visible(False)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.4, left=0.05, right=0.99)
        plt.savefig(filename)
        print(f'Saved {filename}')
        plt.show() if show_plot else plt.close()

    # Store filtered DataFrames
    filtered_dfs = {}

    # Process each HZ limit
    for hz_limit in hz_limits:
        if hz_limit is None:
            # Process all data
            filtered_df = df.copy()
            title = 'All Data'
            suffix = ''
        else:
            # Filter data based on HZ Detection Limit
            filtered_df = df[df['HZ Detection Limit [M_Earth]'] < hz_limit].copy()
            title = f'HZ Detection Limit [M_Earth] < {hz_limit}'
            suffix = f'_M_earth_{str(hz_limit).replace(".", "_")}'

        # Store filtered DataFrame
        filtered_dfs[hz_limit] = filtered_df

        # Create plots
        plot_histograms(
            filtered_df, 
            title, 
            f'{FIGURES_DIRECTORY}star_properties_histograms{suffix}.png'
        )

        # Save to Excel
        output_path = f'{RESULTS_DIRECTORY}Gaia_homogeneous_target_selection{suffix}_{date_str}.xlsx'
        filtered_df.sort_values(
            by='HZ Detection Limit [M_Earth]'
        ).to_excel(output_path, index=False)

        adjust_column_widths(output_path)

    return filtered_dfs

#---------------------------------------------------------------------------------------------------


def plot_scatter_with_options(df, col_x, col_y, min_value=None, max_value=None, label=False, show_plot=False):
    # Create a scatter plot with color mapping
    plt.figure(figsize=(8, 6), dpi=150)
    scatter = plt.scatter(
        df[col_x], 
        df[col_y], 
        c=df['T_eff [K]'], 
        cmap='autumn', 
        edgecolor='k', 
        alpha=0.7
    )

    # Add titles and labels
    # plt.title('Selection Crossmatch', fontsize=14)
    plt.xlabel(f'{col_x} (Ralf)', fontsize=12)
    plt.ylabel(f'{col_y} (Jinglin)', fontsize=12)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('T_eff [K]', fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot the x = y line
    if min_value is None:
        min_value = min(min(df[col_x]), min(df[col_y]))
    if max_value is None:
        max_value = max(max(df[col_x]), max(df[col_y]))
    plt.plot([min_value, max_value], [min_value, max_value], color='gray', linestyle='--')
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)

    if label:
        # Add labels to each point
        x_range = max_value - min_value
        x_offset = x_range * 0.01
        for i, name in enumerate(df['star_ID  ']):
            if (df[col_x][i] > min_value) and (df[col_x][i] < max_value) and (df[col_y][i] > min_value) and (df[col_y][i] < max_value):
                plt.text(df[col_x][i] - x_offset, df[col_y][i], name, fontsize=5, ha='right')    

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIRECTORY}crossmatch_' + col_y.strip().replace(" ", "_").replace("[", "").replace("]", "").replace("/", "") + '.png')
    print(f'Saved {FIGURES_DIRECTORY}crossmatch_' + col_y.strip().replace(" ", "_").replace("[", "").replace("]", "").replace("/", "") + '.png')
    plt.show() if show_plot else plt.close()

#---------------------------------------------------------------------------------------------------

def plot_RV_precision_HZ_detection_limit_vs_temperature(merged_df, df_Ralf):
    print('\nPlotting RV precision and HZ detection limit vs temperature')
    i = 7
    colors = plt.cm.viridis(np.linspace(0, 1, 8))

    # Use the function to create the plot
    plot_scatter(
        x='T_eff [K]',
        y='σ_photon [m/s]',
        data=merged_df,
        xlabel='Stellar Temperature (K)',
        ylabel='σ_photon [m/s]',
        xlim=(min(min(merged_df['T_eff [K]']), min(df_Ralf['Teff '])) - 100, max(max(merged_df['T_eff [K]']), max(df_Ralf['Teff '])) + 100),
        ylim=(0, 2),
        filename=f'{FIGURES_DIRECTORY}RV_precision_vs_temperature.png',
        color=colors[i-1],  # Assuming 'colors' is defined and 'i' is an integer index
        x2 = 'Teff ', 
        y2 = 'RV_Prec(390-870) 30m',
        data2 = df_Ralf,
        color2 = 'red'
    )

    plot_scatter(
        x='T_eff [K]',
        y='HZ Detection Limit [M_Earth]',
        data=merged_df,
        xlabel='Stellar Temperature (K)',
        ylabel='HZ Detection Limit (M_Earth)',
        xlim=(min(merged_df['T_eff [K]']) - 200, 6000 + 500),
        ylim=(0, 10),
        filename=f'{FIGURES_DIRECTORY}HZ_detection_limit_vs_temperature_full.png',
        color=colors[i],  # Replace with actual color if using a list
        x2 = 'Teff ', 
        y2 = 'mdl(hz) 30min',
        data2 = df_Ralf,
        color2 = 'red'
    )

    plot_scatter(
        x='T_eff [K]',
        y='HZ Detection Limit [M_Earth]',
        data=merged_df,
        xlabel='Stellar Temperature (K)',
        ylabel='HZ Detection Limit (M_Earth)',
        xlim=(min(merged_df['T_eff [K]']) - 200, 6000 + 100),
        ylim=(0, 4),
        filename=f'{FIGURES_DIRECTORY}HZ_detection_limit_vs_temperature_zoomed_4.png',
        color=colors[i],  # Replace with actual color if using a list
        x2 = 'Teff ', 
        y2 = 'mdl(hz) 30min',
        data2 = df_Ralf,
        color2 = 'red'
    )

    plot_scatter(
        x='T_eff [K]',
        y='HZ Detection Limit [M_Earth]',
        data=merged_df,
        xlabel='Stellar Temperature (K)',
        ylabel='HZ Detection Limit (M_Earth)',
        xlim=(min(merged_df['T_eff [K]']) - 200, 6000 + 100),
        ylim=(0, 1.5),
        filename=f'{FIGURES_DIRECTORY}HZ_detection_limit_vs_temperature_zoomed_1_5.png',
        color=colors[i],  # Replace with actual color if using a list
        x2 = 'Teff ', 
        y2 = 'mdl(hz) 30min',
        data2 = df_Ralf,
        color2 = 'red'
    )