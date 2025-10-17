#!/usr/bin/env python3
"""
Advanced plotting experiment script for testing modifications to plotting functions.
This script provides an interactive way to experiment with different plotting parameters.
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
    """Load existing processed data or create sample data for testing."""
    print("Loading sample data for plotting experiments...")
    
    # Try to load the most recent processed data
    possible_files = [
        GAIA_FILE,
        backup_file,
        f"{RESULTS_DIRECTORY}merged_df_with_rhk.xlsx",
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
    """Create realistic sample data for testing."""
    np.random.seed(42)
    n_stars = 1000
    
    data = {
        'T_eff [K]': np.random.uniform(4000, 6500, n_stars),
        'Luminosity [L_Sun]': np.random.uniform(0.1, 5.0, n_stars),
        'Radius [R_Sun]': np.random.uniform(0.5, 2.0, n_stars),
        'HZ Detection Limit [M_Earth]': np.random.uniform(0.5, 6.0, n_stars),
        'Mass [M_Sun]': np.random.uniform(0.6, 1.4, n_stars),
        'logg_gaia': np.random.uniform(3.8, 4.8, n_stars),
        'Density [Solar unit]': np.random.uniform(0.1, 5.0, n_stars),
    }
    
    return pd.DataFrame(data)

def test_plotting_variations(df):
    """Test different variations of the plotting function."""
    
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
    
    # Test 1: Original function
    print("\n" + "="*50)
    print("Test 1: Original function")
    print("="*50)
    try:
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=[None, 4, 2, 1.5],
            dpi=150,
            show_plot=True
        )
        print("✅ Original function works!")
    except Exception as e:
        print(f"❌ Original function failed: {e}")
    
    # Test 2: Different detection limits
    print("\n" + "="*50)
    print("Test 2: Different detection limits")
    print("="*50)
    try:
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=[None, 3, 2, 1],
            dpi=150,
            show_plot=True
        )
        print("✅ Different detection limits work!")
    except Exception as e:
        print(f"❌ Different detection limits failed: {e}")
    
    # Test 3: Higher DPI
    print("\n" + "="*50)
    print("Test 3: Higher DPI")
    print("="*50)
    try:
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=[None, 4, 2, 1.5],
            dpi=300,
            show_plot=True
        )
        print("✅ Higher DPI works!")
    except Exception as e:
        print(f"❌ Higher DPI failed: {e}")

def interactive_experiment():
    """Interactive experiment mode."""
    print("\n" + "="*60)
    print("Interactive Plotting Experiment")
    print("="*60)
    
    df = load_sample_data()
    
    while True:
        print(f"\nCurrent data: {len(df)} stars")
        print("HZ Detection Limit range:", 
              f"{df['HZ Detection Limit [M_Earth]'].min():.2f} - {df['HZ Detection Limit [M_Earth]'].max():.2f} M_Earth")
        
        print("\nOptions:")
        print("1. Test original function")
        print("2. Test with custom detection limits")
        print("3. Test with custom DPI")
        print("4. Run all tests")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            test_plotting_variations(df)
        elif choice == '2':
            limits_input = input("Enter detection limits (comma-separated, e.g., None,3,2,1): ").strip()
            try:
                limits = [None if x.strip() == 'None' else float(x.strip()) for x in limits_input.split(',')]
                plot_hr_diagram_multi_detection_limits(df=df, detection_limits=limits, show_plot=True)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '3':
            dpi = int(input("Enter DPI (e.g., 300): "))
            plot_hr_diagram_multi_detection_limits(df=df, dpi=dpi, show_plot=True)
        elif choice == '4':
            test_plotting_variations(df)
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main function."""
    print("2ES Plotting Experiment Script")
    print("=" * 40)
    
    mode = input("Choose mode:\n1. Quick test\n2. Interactive experiment\nEnter choice (1 or 2): ").strip()
    
    df = load_sample_data()
    
    if mode == '1':
        test_plotting_variations(df)
    elif mode == '2':
        interactive_experiment()
    else:
        print("Invalid choice. Running quick test...")
        test_plotting_variations(df)

if __name__ == "__main__":
    main()
