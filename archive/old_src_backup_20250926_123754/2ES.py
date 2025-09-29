print('\nInitializing 2ES Target Selection Pipeline...')

from config import *
from gaia_queries import *
from data_processing import *
from plotting import *
from stellar_calculations import *
from utils import *
from catalog_integration import CatalogProcessor, add_rhk_to_dataframe
from filtering import filter_stellar_data
from gaia_tess_overlap import run_tess_overlap_batch
from HWO_overlap import HWO_match
from plato_lops2 import plato_lops2_match
import matplotlib.pyplot as plt

def main():
    # -----------------------------------------------------------
    # Gaia queries and merging
    # -----------------------------------------------------------
    df_dr2 = execute_gaia_query(
        get_dr2_query(),
        str_columns=['source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr2_results.xlsx"
    )

    dr2_source_ids = tuple(df_dr2['source_id'])
    df_crossmatch = execute_gaia_query(
        get_crossmatch_query(dr2_source_ids),
        str_columns=['dr2_source_id', 'dr3_source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr2_dr3_crossmatch.xlsx"
    )

    df_dr3 = execute_gaia_query(
        get_dr3_query(),
        str_columns=['source_id'],
        output_file=f"{RESULTS_DIRECTORY}dr3_results.xlsx"
    )

    merged_results = process_gaia_data(df_dr2, df_dr3, df_crossmatch)
    clean_results = clean_merged_results(merged_results)
    consolidated_results = consolidate_data(clean_results)

    # -----------------------------------------------------------
    # External catalog enrichment
    # -----------------------------------------------------------
    processor = CatalogProcessor(
        celesta_path='../data/Catalogue_CELESTA.txt',
        stellar_catalog_path='../data/Catalogue_V_117A_table1.txt'
    )
    df_consolidated = processor.process_catalogs(consolidated_results)

    plot_density_vs_logg(
        df_consolidated,
        output_path=f'{FIGURES_DIRECTORY}density_vs_logg.png',
        show_plot=False
    )

    df_consolidated = add_rhk_to_dataframe(df_consolidated)
    save_and_adjust_column_widths(df_consolidated, '../results/merged_df_with_rhk.xlsx')

    # -----------------------------------------------------------
    # Filtering + color-magnitude diagram
    # -----------------------------------------------------------
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

    df_with_bright_neighbors, df_without_bright_neighbors = analyze_bright_neighbors(
        merged_df=combined_df,
        search_radius=SEARCH_RADIUS,
        execute_gaia_query_func=execute_gaia_query
    )

    df = df_without_bright_neighbors.copy()

    # -----------------------------------------------------------
    # Habitable zone & noise modeling
    # -----------------------------------------------------------
    df = calculate_and_insert_habitable_zone(df)
    df = calculate_and_insert_photon_noise(df)
    df = calculate_hz_orbital_period(df)
    df = add_granulation_to_dataframe(df)
    df = add_pmode_rms_to_dataframe(df)
    df = calculate_and_insert_RV_noise(df)
    df = calculate_K(df)
    df = calculate_K(df, sigma_rv_col='σ_RV,total [m/s]')
    df = calculate_and_insert_hz_detection_limit(
        df,
        semi_amplitude_col='K_σ_photon [m/s]'
    )
    df = calculate_and_insert_hz_detection_limit(
        df,
        semi_amplitude_col='K_σ_RV_total [m/s]'
    )

    # -----------------------------------------------------------
    # Plots
    # -----------------------------------------------------------
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

    for detection_limit in DETECTION_LIMITS:
        plot_hr_diagram_with_detection_limit(
            df,
            use_filtered_data=detection_limit is not None,
            detection_limit=detection_limit
        )

    if len(DETECTION_LIMITS) == 4:
        plot_hr_diagram_multi_detection_limits(
            df=df,
            detection_limits=DETECTION_LIMITS,
            show_plot=False
        )

    filtered_dfs = analyze_stellar_data(
        df=df,
        hz_limits=DETECTION_LIMITS,
        show_plot=False
    )

    # -----------------------------------------------------------
    # Comparison with Ralf's results
    # -----------------------------------------------------------
    merged_RJ, df_Ralf = merge_and_format_stellar_data(
        df_main=df,
        ralf_file_path=RALF_FILE_PATH
    )
    merged_RJ['HZ Rmid'] = (merged_RJ['HZ Rin'] + merged_RJ['HZ Rout']) / 2

    plot_scatter_with_options(merged_RJ, 'magV     ', 'V_mag', min_value=3, max_value=10)
    plot_scatter_with_options(merged_RJ, 'mass ', 'Mass [M_Sun]', min_value=0.5, max_value=1.5)
    plot_scatter_with_options(merged_RJ, 'HZ Rmid', 'HZ_limit [AU]', min_value=0.1, max_value=2, label=True)
    plot_scatter_with_options(merged_RJ, 'logg', 'logg_gaia', min_value=1.9, max_value=5, label=True)
    plot_scatter_with_options(merged_RJ, 'RV_Prec(390-870) 30m', 'σ_photon [m/s]', min_value=0, max_value=1.6, label=True)
    plot_scatter_with_options(merged_RJ, 'mdl(hz) 30min', 'HZ Detection Limit [M_Earth]', min_value=0, max_value=3, label=True)

    plot_RV_precision_HZ_detection_limit_vs_temperature(df, df_Ralf)

    print("\nRalf's results:")
    print("Number of stars:", len(merged_RJ))
    for detection_limit in DETECTION_LIMITS:
        if detection_limit is not None:
            count_ralf = (merged_RJ['mdl(hz) 30min'] < detection_limit).sum()
            print(f"Number with HZ Detection Limit [M_Earth] < {detection_limit}: {count_ralf}")

    # -----------------------------------------------------------
    # HWO cross-matching
    # -----------------------------------------------------------
    df = HWO_match(df)

    # -----------------------------------------------------------
    # Plato Lops2 cross-matching
    # -----------------------------------------------------------
    df = plato_lops2_match(df)

    # -----------------------------------------------------------
    # TESS cross-matching (confirmed + candidates)
    # -----------------------------------------------------------
    tess_results = run_tess_overlap_batch(df)
    confirmed_gaia_ids = tess_results["confirmed"]["gaia_ids"]
    candidate_gaia_ids = tess_results["candidates"]["gaia_ids"]
    df = tess_results["df"]

    print(f"\nConfirmed TESS host GAIA IDs matched: {len(confirmed_gaia_ids)}")
    print(f"TESS candidate host GAIA IDs matched: {len(candidate_gaia_ids)}")

    df['sum_score'] = (
        df['HWO_match'] * 1 +
        df['LOPS2_match'] * 1 +
        df['TESS_confirmed_match'] * 1 +
        df['TESS_candidate_match'] * 0.5
    )
    df = df.sort_values(['sum_score', 'HZ Detection Limit [M_Earth]'], ascending=[False, True])
    df = df[df['HZ Detection Limit [M_Earth]'] <= 2]
    # df = df[df['HZ Detection Limit [M_Earth]'] <= 4]
    save_and_adjust_column_widths(df, GAIA_FILE)

if __name__ == "__main__":
    main()