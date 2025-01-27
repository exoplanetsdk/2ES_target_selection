import matplotlib.pyplot as plt
import seaborn as sns
from config import *

def plot_hr_diagram(df, filename=None):
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
    # Rest of your plotting code...

def plot_stellar_properties(df, detection_limit):
    # Your existing plotting code...

# Add other plotting functions...
