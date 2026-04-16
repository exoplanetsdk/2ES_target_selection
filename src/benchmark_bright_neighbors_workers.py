"""
Benchmark max_workers for analyze_bright_neighbors to find the sweet spot
(speed vs Gaia archive 500 errors). Run from project root:

  python src/benchmark_bright_neighbors_workers.py

Uses 80 stars and tests max_workers in [4, 6, 8, 10, 12, 16, 20]; reports
time and stars/s, then suggests the best value. Typical sweet spot: 8 (safe)
to 10–12 (faster; may hit 500s under load). Requires results/consolidated_results.xlsx
or results/merged_df_with_rhk.xlsx from a prior pipeline run.
"""
import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Run from src/ so config and utils resolve
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.getcwd()) != 'src':
    os.chdir(_script_dir)
    sys.path.insert(0, _script_dir)

from config import RESULTS_DIRECTORY, SEARCH_RADIUS
from core.utils import execute_gaia_query

# Suppress noisy logs during benchmark
import logging
logging.getLogger('astroquery').setLevel(logging.WARNING)
logging.getLogger('astropy').setLevel(logging.WARNING)


def load_sample_df(n_rows=80):
    """Load a subset of consolidated/merged data for benchmarking."""
    for name in ('consolidated_results.xlsx', 'merged_df_with_rhk.xlsx'):
        path = os.path.join(RESULTS_DIRECTORY, name)
        if os.path.isfile(path):
            df = pd.read_excel(path, nrows=n_rows * 2)
            df = df.head(n_rows)
            if 'ra' in df.columns and 'RA' not in df.columns:
                df = df.rename(columns={'ra': 'RA', 'dec': 'DEC'})
            if 'phot_g_mean_mag' in df.columns and 'Phot G Mean Mag' not in df.columns:
                df = df.rename(columns={'phot_g_mean_mag': 'Phot G Mean Mag'})
            required = {'RA', 'DEC', 'Phot G Mean Mag', 'source_id_dr2'}
            if not required.issubset(df.columns):
                raise SystemExit(f"Missing columns in {path}. Need {required}; got {list(df.columns)[:15]}...")
            return df
    raise SystemExit(
        f"No data file found in {os.path.abspath(RESULTS_DIRECTORY)}. "
        "Run the pipeline once to create consolidated_results.xlsx or merged_df_with_rhk.xlsx."
    )


def run_bright_neighbor_benchmark(merged_df, search_radius, max_workers, max_retries=3, delay=5):
    """Inline version of analyze_bright_neighbors for benchmarking (no data_processing import)."""
    def create_neighbor_query(source_id, ra, dec, neighbor_g_mag_limit, search_radius, data_release):
        return f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM {data_release}.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', {ra}, {dec}), CIRCLE('ICRS', ra, dec, {search_radius}))
          AND phot_g_mean_mag < {neighbor_g_mag_limit} AND source_id != {source_id}
        """

    def process_row(row):
        for attempt in range(max_retries):
            try:
                if not pd.isna(row.get('source_id_dr3')):
                    query = create_neighbor_query(
                        row['source_id_dr3'], row['RA'], row['DEC'],
                        row['Phot G Mean Mag'] + 6.5, search_radius, 'gaiadr3')
                else:
                    query = create_neighbor_query(
                        row['source_id_dr2'], row['RA'], row['DEC'],
                        row['Phot G Mean Mag'] + 6.5, search_radius, 'gaiadr2')
                neighbors_df = execute_gaia_query(query)
                return (row, neighbors_df is not None and not neighbors_df.empty)
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(delay)
        return (row, False)

    rows = [row for _, row in merged_df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_row, rows),
            total=len(rows),
            desc=f"max_workers={max_workers}",
            ncols=80,
            leave=False,
        ))
    return results


def main():
    n_rows = 80
    workers_to_test = [4, 6, 8, 10, 12, 16, 20]

    print("Loading sample data...")
    sample_df = load_sample_df(n_rows=n_rows)
    print(f"Benchmarking with {len(sample_df)} stars and max_workers in {workers_to_test}\n")

    results = []
    for max_workers in workers_to_test:
        start = time.perf_counter()
        try:
            run_bright_neighbor_benchmark(
                sample_df, SEARCH_RADIUS, max_workers=max_workers
            )
            elapsed = time.perf_counter() - start
            rate = len(sample_df) / elapsed
            results.append((max_workers, elapsed, rate, None))
            print(f"  max_workers={max_workers}: {elapsed:.1f}s ({rate:.2f} stars/s)")
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((max_workers, elapsed, 0, str(e)))
            print(f"  max_workers={max_workers}: FAILED after {elapsed:.1f}s - {e}")

    print("\n" + "=" * 60)
    print("Summary (extrapolated to 1361 stars)")
    print("=" * 60)
    ok = [(w, t, r) for w, t, r, err in results if err is None]
    if not ok:
        print("No successful runs.")
        return
    best = max(ok, key=lambda x: x[2])
    print(f"{'max_workers':<12} {'Time (80 stars)':<18} {'Stars/s':<10} {'Est. time (1361)':<18}")
    print("-" * 60)
    for max_workers, elapsed, rate in ok:
        est_full = (1361 / len(sample_df)) * elapsed
        print(f"{max_workers:<12} {elapsed:<18.1f} {rate:<10.2f} {est_full/60:.1f} min")
    print("-" * 60)
    print(f"Sweet spot: max_workers={best[0]} (best throughput: {best[2]:.2f} stars/s)")
    print("If you see HTTP 500 errors in production, use a lower max_workers (e.g. 6 or 8).")


if __name__ == "__main__":
    main()
