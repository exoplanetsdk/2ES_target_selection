from astroquery.gaia import Gaia
from config import *
from utils import execute_gaia_query

def get_dr2_query():
    return f"""
    SELECT 
        gs.source_id, 
        gs.ra, 
        gs.dec, 
        gs.phot_g_mean_mag, 
        gs.phot_bp_mean_mag, 
        gs.phot_rp_mean_mag,
        gs.parallax,
        COALESCE(ap_supp.teff_gspphot_marcs, ap.teff_gspphot, gs.teff_val) AS teff_val,
        COALESCE(ap_supp.mass_flame_spec, ap.mass_flame) AS mass_flame,
        COALESCE(ap_supp.lum_flame_spec, ap.lum_flame, gs.lum_val) AS lum_flame,
        COALESCE(ap_supp.radius_flame_spec, ap.radius_flame, gs.radius_val) AS radius_flame,
        COALESCE(ap_supp.logg_gspphot_marcs, ap.logg_gspphot) AS logg_gaia,
        ap.spectraltype_esphs
    FROM 
        gaiadr2.gaia_source AS gs
    LEFT JOIN 
        gaiadr3.astrophysical_parameters AS ap 
        ON gs.source_id = ap.source_id
    LEFT JOIN 
        gaiadr3.astrophysical_parameters_supp AS ap_supp 
        ON gs.source_id = ap_supp.source_id
    WHERE 
        gs.phot_g_mean_mag < {TARGET_G_MAG_LIMIT}
        AND gs.dec BETWEEN {MIN_DEC} AND {MAX_DEC}
        AND gs.parallax >= {MIN_PARALLAX}
    """

def get_dr3_query():
    return f"""
    SELECT 
        gs.source_id, 
        gs.ra, 
        gs.dec, 
        gs.phot_g_mean_mag, 
        gs.phot_bp_mean_mag, 
        gs.phot_rp_mean_mag,
        gs.bp_rp, 
        gs.parallax,
        COALESCE(ap_supp.teff_gspphot_marcs, ap.teff_gspphot, gs.teff_gspphot) AS teff_gspphot,
        COALESCE(ap_supp.mass_flame_spec, ap.mass_flame) AS mass_flame,
        COALESCE(ap_supp.lum_flame_spec, ap.lum_flame) AS lum_flame,
        COALESCE(ap_supp.radius_flame_spec, ap.radius_flame) AS radius_flame,
        COALESCE(ap_supp.logg_gspphot_marcs, ap.logg_gspphot) AS logg_gaia,
        ap.spectraltype_esphs
    FROM 
        gaiadr3.gaia_source AS gs
    LEFT JOIN 
        gaiadr3.astrophysical_parameters AS ap 
        ON gs.source_id = ap.source_id
    LEFT JOIN 
        gaiadr3.astrophysical_parameters_supp AS ap_supp 
        ON gs.source_id = ap_supp.source_id
    WHERE 
        gs.phot_g_mean_mag < {TARGET_G_MAG_LIMIT}
        AND gs.dec BETWEEN {MIN_DEC} AND {MAX_DEC}
        AND gs.parallax >= {MIN_PARALLAX}
    """

def get_crossmatch_query(dr2_source_ids):
    return f"""
    SELECT dr2_source_id, dr3_source_id
    FROM gaiadr3.dr2_neighbourhood
    WHERE dr2_source_id IN {dr2_source_ids}
    """
