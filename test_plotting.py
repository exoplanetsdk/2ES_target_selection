#!/usr/bin/env python3
"""
Test script for experimenting with plotting functions without running the full 2ES pipeline.
This script loads existing processed data and allows you to test plotting modifications quickly.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from analysis.plotting import plot_hr_diagram_multi_detection_limits

def load_sample_data():
    """
    Load existing processed data for testing.
    Try to load the most recent processed data file.
    """
    print("Loading sample data for plotting experiments...")
    
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
        print("No processed data found. Creating sample data for testing...")
        df = create_sample_data()
    
    return df

def create_sample_data():
    """
    Create sample data for testing if no processed data is available.
    """
    np.random.seed(42)  # For reproducible results
    
    n_stars = 1000
    
    # Create realistic stellar data
    data = {
        'T_eff [K]': np.random.uniform(4000, 6500, n_stars),
        'Luminosity [L_Sun]': np.random.uniform(0.1, 5.0, n_stars),
        'Radius [R_Sun]': np.random.uniform(0.5, 2.0, n_stars),
        'HZ Detection Limit [M_Earth]': np.random.uniform(0.5, 6.0, n_stars),
        'Mass [M_Sun]': np.random.uniform(0.6, 1.4, n_stars),
        'logg_gaia': np.random.uniform(3.8, 4.8, n_stars),
        'Density [Solar unit]': np.random.uniform(0.1, 5.0, n_stars),
    }
    
    df = pd.DataFrame(data)
    print(f"Created sample data with {len(df)} stars")
    return df

def test_plotting_function(df, detection_limits=[None, 4, 2, 1.5]):
    """
    Test the plotting function with different parameters.
    """
    print(f"\nTesting plot_hr_diagram_multi_detection_limits with {len(df)} stars")
    print(f"Detection limits: {detection_limits}")
    
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
    
    try:
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=detection_limits,
            dpi=150,
            show_plot=True  # Show the plot for immediate feedback
        )
        print("✅ Plotting function executed successfully!")
    except Exception as e:
        print(f"❌ Error in plotting function: {e}")
        import traceback
        traceback.print_exc()

def experiment_with_plotting():
    """
    Main function to experiment with plotting modifications.
    """
    print("=" * 60)
    print("2ES Plotting Experiment Script")
    print("=" * 60)
    
    # Load data
    df = load_sample_data()
    
    # Display basic info about the data
    print(f"\nData summary:")
    print(f"Number of stars: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"HZ Detection Limit range: {df['HZ Detection Limit [M_Earth]'].min():.2f} - {df['HZ Detection Limit [M_Earth]'].max():.2f} M_Earth")
    
    # Test the original function
    print(f"\n{'='*40}")
    print("Testing original plotting function...")
    print(f"{'='*40}")
    test_plotting_function(df)
    
    # You can add more experiments here
    print(f"\n{'='*40}")
    print("Ready for your experiments!")
    print("Modify the plot_hr_diagram_multi_detection_limits function in src/analysis/plotting.py")
    print("Then run this script again to see your changes.")
    print(f"{'='*40}")

if __name__ == "__main__":
    experiment_with_plotting()
