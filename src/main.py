print('\nInitializing 2ES Target Selection Pipeline...')

from config import *
from gaia_queries import *
from data_processing import *
from plotting import *
from stellar_calculations import *
from utils import *
from catalog_integration import CatalogProcessor
from filtering import filter_stellar_data
from gaia_tess_overlap import match_gaia_tess, save_overlapping_stars

def main():

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
    consolidated_results = consolidate_data(clean_results)

    # Obtain stellar properties from catalogs
    processor = CatalogProcessor(
        celesta_path ='../data/Catalogue_CELESTA.txt',
        stellar_catalog_path ='../data/Catalogue_V_117A_table1.txt'
    )
    df_consolidated = processor.process_catalogs(consolidated_results)

    # Plot density vs logg
    plot_density_vs_logg(
        df_consolidated, 
        output_path=f'{FIGURES_DIRECTORY}density_vs_logg.png', 
        show_plot=False
    )

    df_filtered = filter_stellar_data(df_consolidated, STELLAR_FILTERS)

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

    df_with_hz                  = calculate_and_insert_habitable_zone(combined_df)
    df_with_rv_precision        = calculate_and_insert_rv_precision(df_with_hz)
    df_with_hz_orbital_period   = calculate_hz_orbital_period(df_with_rv_precision)
    df_with_K                   = calculate_K(df_with_hz_orbital_period)
    merged_df                   = calculate_and_insert_hz_detection_limit(df_with_K)

    df_with_bright_neighbors, df_without_bright_neighbors = analyze_bright_neighbors(
        merged_df=merged_df,
        search_radius=SEARCH_RADIUS,
        execute_gaia_query_func=execute_gaia_query
    )

    merged_df = df_without_bright_neighbors.copy()

    # Add granulation noise and p-mode noise
    merged_df = add_granulation_to_dataframe(merged_df)
    merged_df = add_pmode_rms_to_dataframe(merged_df)

    #---------------------------------------------------------------------------------------------------    
    # Plotting
    #---------------------------------------------------------------------------------------------------    

    # Plot RA vs DEC
    plot_scatter(
        x='RA',
        y='DEC',
        data=merged_df,
        xlabel='Right Ascension (RA)',
        ylabel='Declination (DEC)',
        xlim=(0, 360),
        filename=f'{FIGURES_DIRECTORY}ra_dec.png',
        alpha=0.6,
        invert_xaxis=True,
        show_plot=False
    )

    # Plot HR diagram
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(
        merged_df['T_eff [K]'], 
        merged_df['Luminosity [L_Sun]'], 
        c=merged_df['T_eff [K]'],  # Color by temperature
        cmap='autumn',  # Use autumn colormap for red to yellow transition
        alpha=0.99, 
        edgecolors='w',  # Use white for edges
        linewidths=0.05,  # Set edge width
        s=merged_df['Radius [R_Sun]'] * 20  # Scale the radius for visibility
    )
    plt.colorbar(label='Effective Temperature (K)')  # Add color bar
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(merged_df['T_eff [K]']) - 50, max(merged_df['T_eff [K]']) + 50)  # Set the same x range
    plt.ylim(min(merged_df['Luminosity [L_Sun]']), max(merged_df['Luminosity [L_Sun]']) + 0.5)  # Set the same y range
    plt.gca().invert_xaxis()  # Invert x-axis for temperature
    plt.xlabel('Effective Temperature (K)')
    plt.ylabel('Luminosity (L/L_sun)')
    plt.title('Hertzsprung-Russell Diagram')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig(f'{FIGURES_DIRECTORY}HR_diagram.png')
    plt.close()
    
    # Plot HR diagram with detection limit
    for detection_limit in DETECTION_LIMITS:
        plot_hr_diagram_with_detection_limit(
            merged_df, 
            use_filtered_data=detection_limit is not None, 
            detection_limit=detection_limit
        )
        
    if len(DETECTION_LIMITS) == 4:
        plot_hr_diagram_multi_detection_limits(
            df=merged_df,
            detection_limits=DETECTION_LIMITS,
            show_plot=False
        )

    filtered_dfs = analyze_stellar_data(
        df=merged_df,
        hz_limits=DETECTION_LIMITS,
        show_plot=False
    )

    #---------------------------------------------------------------------------------------------------    
    # Comparison with Ralf's results
    #---------------------------------------------------------------------------------------------------    
    merged_RJ, df_Ralf = merge_and_format_stellar_data(
        df_main=merged_df,
        ralf_file_path=RALF_FILE_PATH
    )

    merged_RJ['HZ Rmid'] = (merged_RJ['HZ Rin'] + merged_RJ['HZ Rout']) / 2
    plot_scatter_with_options(merged_RJ, 'magV     ', 'V_mag', min_value=3, max_value=10)
    plot_scatter_with_options(merged_RJ, 'mass ', 'Mass [M_Sun]', min_value=0.5, max_value=1.5)
    plot_scatter_with_options(merged_RJ, 'HZ Rmid', 'HZ_limit [AU]', min_value=0.1, max_value=2, label=True)
    plot_scatter_with_options(merged_RJ, 'logg', 'logg_gaia', min_value=1.9, max_value=5, label=True)
    plot_scatter_with_options(merged_RJ, 'RV_Prec(390-870) 30m', 'RV precision [m/s]', min_value=0, max_value=1.6, label=True)
    plot_scatter_with_options(merged_RJ, 'mdl(hz) 30min', 'HZ Detection Limit [M_Earth]', min_value=0, max_value=3, label=True)

    plot_RV_precision_HZ_detection_limit_vs_temperature(merged_df, df_Ralf)

    print("\nRalf's results:")
    print("Number of stars:", len(merged_RJ))
    for detection_limit in DETECTION_LIMITS:
        if detection_limit is not None:
            print(f"Number of stars with HZ Detection Limit [M_Earth] < {detection_limit}:", len(merged_RJ[merged_RJ['mdl(hz) 30min'] < detection_limit]))

    #---------------------------------------------------------------------------------------------------    
    # TESS overlap
    #---------------------------------------------------------------------------------------------------    
    # Process confirmed planets
    print("\nProcessing confirmed TESS planets...")
    matches_confirmed, merged_confirmed, confirmed_gaia_ids = match_gaia_tess(
        GAIA_FILE,
        TESS_CONFIRMED_FILE,
        OUTPUT_CONFIRMED_FILE,
        OUTPUT_CONFIRMED_UNIQUE_PLANETS,
        is_candidate=False,
        threshold_arcsec=2.5
    )
    # Save overlapping stars for confirmed planets
    overlapping_stars = save_overlapping_stars(GAIA_FILE, confirmed_gaia_ids, OUTPUT_CONFIRMED_UNIQUE_STARS)

    # Process candidates
    print("\nProcessing TESS candidates...")
    matches_candidates, merged_candidates, candidate_gaia_ids = match_gaia_tess(
        GAIA_FILE,
        TESS_CANDIDATE_FILE,
        OUTPUT_CANDIDATE_FILE,
        OUTPUT_CANDIDATE_UNIQUE_PLANETS,
        is_candidate=True,
        threshold_arcsec=2.5
    )
    # Save overlapping stars for candidates only
    overlapping_stars = save_overlapping_stars(GAIA_FILE, candidate_gaia_ids, OUTPUT_CANDIDATE_UNIQUE_STARS)


if __name__ == "__main__":
    main()

