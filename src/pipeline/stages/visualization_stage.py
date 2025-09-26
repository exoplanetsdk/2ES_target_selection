"""
Visualization stage - implements the original plotting functionality.
"""
import pandas as pd
import matplotlib.pyplot as plt
from ..base_simple import PipelineStage
from core.exceptions import PipelineError
from . import import_helpers  # This sets up the archive imports
from plotting import (
    plot_scatter,
    plot_hr_diagram_with_detection_limit,
    plot_hr_diagram_multi_detection_limits,
    analyze_stellar_data
)


class VisualizationStage(PipelineStage):
    """Pipeline stage for creating visualizations."""
    
    def process(self, data):
        """Create all the plots and visualizations."""
        self.logger.info("Starting visualization generation")
        
        try:
            df = data.copy()
            
            # Create RA/Dec plot
            self.logger.info("Creating RA/Dec plot")
            plot_scatter(
                x='RA',
                y='DEC',
                data=df,
                xlabel='Right Ascension (RA)',
                ylabel='Declination (DEC)',
                xlim=(0, 360),
                filename=f'{self.config.paths.figures_dir}/ra_dec.png',
                alpha=0.6,
                invert_xaxis=True,
                show_plot=False
            )
            
            # Create HR diagram
            self.logger.info("Creating HR diagram")
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
            plt.savefig(f'{self.config.paths.figures_dir}/HR_diagram.png')
            plt.close()
            
            # Create HR diagrams with detection limits
            self.logger.info("Creating HR diagrams with detection limits")
            for detection_limit in self.config.detection_limits:
                plot_hr_diagram_with_detection_limit(
                    df,
                    use_filtered_data=detection_limit is not None,
                    detection_limit=detection_limit
                )
            
            # Create multi-detection limit plot
            if len(self.config.detection_limits) == 4:
                self.logger.info("Creating multi-detection limit plot")
                plot_hr_diagram_multi_detection_limits(
                    df=df,
                    detection_limits=self.config.detection_limits,
                    show_plot=False
                )
            
            # Analyze stellar data
            self.logger.info("Analyzing stellar data")
            filtered_dfs = analyze_stellar_data(
                df=df,
                hz_limits=self.config.detection_limits,
                show_plot=False
            )
            
            self.logger.info(f"Visualization completed. {len(df)} stars visualized")
            return df
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise PipelineError(f"Visualization failed: {e}") from e
