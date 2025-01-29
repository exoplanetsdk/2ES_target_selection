from config import *
from gaia_queries import *
from data_processing import *
from plotting import *
from stellar_calculations import *
from utils import *
from catalog_integration import CatalogProcessor
from filtering import filter_stellar_data

def main():
    # Initialize Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # Execute DR2 query
    df_dr2 = execute_gaia_query(
        get_dr2_query(),
        str_columns=['source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr2_results.xlsx"
    )

    # Execute crossmatch query
    dr2_source_ids = tuple(df_dr2['source_id'])
    df_crossmatch = execute_gaia_query(
        get_crossmatch_query(dr2_source_ids),
        str_columns=['dr2_source_id', 'dr3_source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr2_dr3_crossmatch.xlsx"
    )

    # Execute DR3 query
    df_dr3 = execute_gaia_query(
        get_dr3_query(),
        str_columns=['source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr3_results.xlsx"
    )

    # Process and clean data
    merged_results = process_gaia_data(df_dr2, df_dr3, df_crossmatch)
    clean_results = clean_merged_results(merged_results)
    consolidate_results = consolidate_data(clean_results)


    # Obtain stellar properties from catalogs
    processor = CatalogProcessor(
        celesta_path='../data/Catalogue_CELESTA.txt',
        stellar_catalog_path='../data/Catalogue_V_117A_table1.txt'
    )
    df_consolidated = processor.process_catalogs(consolidate_results)

    # Plot density vs logg
    plot_density_vs_logg(df_consolidated, output_path=f'{FIGURES_DIRECTORY}density_vs_logg.png', show_plot=False)

    df_filtered, _, _ = filter_stellar_data(df_consolidated, STELLAR_FILTERS)

    plot_color_histogram(
        df_filtered,
        output_path=f'{FIGURES_DIRECTORY}color_histogram.png',
        show_plot=False
    )

    combined_df = plot_color_magnitude_diagram(
        df_filtered,
        output_path=f'{FIGURES_DIRECTORY}color_magnitude_diagram.png',
        show_plot=False
    )

    combined_df = calculate_and_insert_habitable_zone(
        combined_df,
        RESULTS_DIRECTORY,
        output_filename='combined_query.xlsx'
    )




    import importlib
    import config
    importlib.reload(config)
    from filtering import filter_stellar_data


    # Generate plots and save results
    plot_hr_diagram(final_results)
    plot_stellar_properties(final_results, detection_limit=1.5)
    
    # Save final results
    final_results.to_excel(f"{RESULTS_DIRECTORY}final_results.xlsx", index=False)

if __name__ == "__main__":
    main()







# In main.py, add:


def main():
    # ... (previous code) ...
    
    # Initialize catalog processor

