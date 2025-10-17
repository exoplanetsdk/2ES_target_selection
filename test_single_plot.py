#!/usr/bin/env python3
"""
Simple script to test just the plot_hr_diagram_multi_detection_limits function.
Perfect for quick experimentation with plotting modifications.
"""

import sys
import os
import pandas as pd

# Add src directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from analysis.plotting import plot_hr_diagram_multi_detection_limits

def load_data():
    """Load the most recent processed data."""
    possible_files = [
        GAIA_FILE,
        backup_file,
        f"{RESULTS_DIRECTORY}merged_df_with_rhk.xlsx",
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading data from: {file_path}")
            return pd.read_excel(file_path)
    
    raise FileNotFoundError("No processed data found. Run 2ES.py first or use test_plotting.py")

def main():
    """Test the specific plotting function."""
    print("Testing plot_hr_diagram_multi_detection_limits function")
    print("=" * 50)
    
    try:
        # Load data
        df = load_data()
        print(f"Loaded {len(df)} stars")
        
        # Test the function
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=DETECTION_LIMITS,
            dpi=150,
            show_plot=True  # Show plot for immediate feedback
        )
        
        print("✅ Function executed successfully!")
        print("Modify the function in src/analysis/plotting.py and run this script again to test changes.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
