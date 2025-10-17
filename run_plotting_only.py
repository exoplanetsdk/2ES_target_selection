#!/usr/bin/env python3
"""
Script to run only the plotting part of the 2ES pipeline.
This allows you to test plotting modifications without running the entire pipeline.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from analysis.plotting import *

def load_processed_data():
    """
    Load the most recent processed data from the 2ES pipeline.
    This assumes the pipeline has been run at least once.
    """
    print("Loading processed data for plotting...")
    
    # Try to load the most recent processed data
    possible_files = [
        GAIA_FILE,  # Today's file from config
        backup_file,  # Backup file from config
        f"{RESULTS_DIRECTORY}merged_df_with_rhk.xlsx",  # Intermediate file
    ]
    
    df = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                print(f"Loading data from: {file_path}")
                df = pd.read_excel(file_path)
                print(f"Successfully loaded {len(df)} stars")
                break
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue
    
    if df is None:
        raise FileNotFoundError(
            "No processed data found. Please run the full 2ES.py pipeline first "
            "to generate the processed data files."
        )
    
    return df

def run_plotting_section(df):
    """
    Run only the plotting section from 2ES.py.
    This is the exact same code as in the main pipeline.
    """
    print("\n" + "="*60)
    print("Running Plotting Section Only")
    print("="*60)
    
    # -----------------------------------------------------------
    # Plots (copied from 2ES.py lines 108-164)
    # -----------------------------------------------------------
    
    # RA/DEC plot
    print("Creating RA/DEC plot...")
    plot_scatter(
        x='RA',
        y='DEC',
        data=df,
        xlabel='Right Ascension (RA)',
        ylabel='Declination (DEC)',
        xlim=(0, 360),
        filename=f'{FIGURES_DIRECTORY}ra_dec.png',
        alpha=0.6,
        invert_xaxis=True,
        show_plot=False
    )
    
    # Basic HR diagram
    print("Creating basic HR diagram...")
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(
        df['T_eff [K]'],
        df['Luminosity [L_Sun]'],
        c=df['T_eff [K]'],
        cmap='autumn',
        alpha=0.99,
        edgecolors='w',
        linewidths=0.05,
        s=df['Radius [R_Sun]'] * 20
    )
    plt.colorbar(label='Effective Temperature (K)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(df['T_eff [K]']) - 50, max(df['T_eff [K]']) + 50)
    plt.ylim(min(df['Luminosity [L_Sun]']), max(df['Luminosity [L_Sun]']) + 0.5)
    plt.gca().invert_xaxis()
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Luminosity (L/L_sun)')
    plt.title('Hertzsprung-Russell Diagram')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig(f'{FIGURES_DIRECTORY}HR_diagram.png')
    plt.close()
    
    # HR diagrams with detection limits
    print("Creating HR diagrams with detection limits...")
    for detection_limit in DETECTION_LIMITS:
        plot_hr_diagram_with_detection_limit(
            df,
            use_filtered_data=detection_limit is not None,
            detection_limit=detection_limit
        )
    
    # Multi-detection limit HR diagram (this is the one you want to experiment with!)
    if len(DETECTION_LIMITS) == 4:
        print("Creating multi-detection limit HR diagram...")
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=DETECTION_LIMITS,
            show_plot=True  # Show the plot for immediate feedback
        )
    
    # Stellar data analysis plots
    print("Creating stellar data analysis plots...")
    filtered_dfs = analyze_stellar_data(
        df=df,
        hz_limits=DETECTION_LIMITS,
        show_plot=False
    )
    
    print("\n✅ All plotting functions completed successfully!")
    print(f"Check the {FIGURES_DIRECTORY} directory for output files.")

def main():
    """Main function to run plotting only."""
    print("2ES Plotting-Only Script")
    print("=" * 40)
    print("This script runs only the plotting section of the 2ES pipeline.")
    print("Modify the plotting functions in src/analysis/plotting.py and run this script to test changes.")
    print("=" * 40)
    
    try:
        # Load processed data
        df = load_processed_data()
        
        # Display data summary
        print(f"\nData summary:")
        print(f"Number of stars: {len(df)}")
        print(f"HZ Detection Limit range: {df['HZ Detection Limit [M_Earth]'].min():.2f} - {df['HZ Detection Limit [M_Earth]'].max():.2f} M_Earth")
        print(f"Detection limits to plot: {DETECTION_LIMITS}")
        
        # Run plotting section
        run_plotting_section(df)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Run the full 2ES.py pipeline first to generate processed data")
        print("2. Or use the test_plotting.py script which creates sample data")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
