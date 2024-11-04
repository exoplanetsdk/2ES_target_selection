"""
# sample_select_gaia.py

# Code to query Gaia DR2 for nearest M-dwarfs

@author: mhobson
"""

# =============================================================================
# Imports and set up
# =============================================================================

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import wget
from datetime import datetime
import glob
from astropy.time import Time
plt.rc('text', usetex=True)
plt.ion()

make_plots = False

# to query archive directly
from astroquery.eso import Eso
eso = Eso()
# set eso query row limit - setting to -1 doesn't work properly!
eso.ROW_LIMIT = 100000

# distance or magnitude query
# 'dist' or 'mag'
query_type = 'mag'

# Note this doesn't do much - catalog is reset in the query function
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Select Data Release 3

# def send_radial_gaia_query(query_size=1000000, distance=10, **kwargs):
#     """
#     Sends an archive query for d < dist, with additional filters taken from
#     Gaia Data Release 2: Observational Hertzsprung-Russell diagrams (Sect. 2.1)
#     Gaia Collaboration, Babusiaux et al. (2018)
#     (https://doi.org/10.1051/0004-6361/201832843)
# 
#     NOTE: 10000000 is a maximum query size (~76 MB / column)
# 
#     Additional keyword arguments are passed to TapPlus.launch_job_async method.
#     """
#     # from astroquery.utils.tap.core import TapPlus
# 
#     # gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")
# 
#     job = Gaia.launch_job_async("select top {}".format(query_size)+
#                 #" lum_val, teff_val,"
#                 #" ra, dec, parallax,"
#                 " bp_rp, phot_g_mean_mag+5*log10(parallax)-10 as mg"
#          " from gaiadr2.gaia_source"
#          " where parallax_over_error > 10"
#          " and visibility_periods_used > 8"
#          " and phot_g_mean_flux_over_error > 50"
#          " and phot_bp_mean_flux_over_error > 20"
#          " and phot_rp_mean_flux_over_error > 20"
#          " and phot_bp_rp_excess_factor <"
#             " 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
#          " and phot_bp_rp_excess_factor >"
#             " 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
#          " and astrometric_chi2_al/(astrometric_n_good_obs_al-5)<"
#             "1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))"
#          +" and 1000/parallax <= {}".format(distance), **kwargs)
# 
#     return job

def send_radial_gaia_query(query_size=1000000, parallax_limit=100, **kwargs):
    """
    Sends an archive query for d < dist

    NOTE: 10000000 is a maximum query size (~76 MB / column)

    """
    # define job
    job = Gaia.launch_job_async("select top {}".format(query_size)+
    	# astrometry
    	" ra, ra_error, dec, dec_error, parallax, parallax_error,"
    	# photometry
    	" phot_g_mean_mag, bp_rp, "
    	# stellar parameters
        # DR2
    	#" teff_val, lum_val, radius_val,"
        # DR3
        " teff_gspphot,"
        # non single star flag
        " non_single_star,"
        # spectral line broadening parameter
        " vbroad,"
    	# astrometry parameters - for filtering binaries
    	" astrometric_gof_al, astrometric_excess_noise, astrometric_excess_noise_sig,"
        # gaia id!
        "source_id"
    	# catalog
        " from gaiadr3.gaia_source"
        # conditions
        " where parallax >= {}".format(parallax_limit), **kwargs)

    # output the table
    return job

def send_radial_gaia_query_gmag(query_size=1000000, gmag_limit=10, **kwargs):
    """
    Sends an archive query for Gmag limit

    NOTE: 10000000 is a maximum query size (~76 MB / column)

    """
    # define job
    job = Gaia.launch_job_async("select top {}".format(query_size)+
        # astrometry
        " ra, ra_error, dec, dec_error, parallax, parallax_error,"
        # photometry
        " phot_g_mean_mag, bp_rp, "
        # stellar parameters
        # DR2
        #" teff_val, lum_val, radius_val,"
        # DR3
        " teff_gspphot,"
        # non single star flag
        " non_single_star,"
        # spectral line broadening parameter
        " vbroad,"
        # astrometry parameters - for filtering binaries
        " astrometric_gof_al, astrometric_excess_noise, astrometric_excess_noise_sig,"
        # gaia id!
        "source_id"
        # catalog
        " from gaiadr3.gaia_source"
        # conditions
        " where phot_g_mean_mag <= {}".format(gmag_limit), **kwargs)

    # output the table
    return job


# =============================================================================
# Query - distance limit
# =============================================================================

# set distance in parsecs
dist = 20

# convert to parallax condition in miliarcsec
parallax_limit = 1000/dist

# set query size
qsize = 50000

# check if query already saved
try:
	# if it is, read it in from file
    gaiarec_d = np.recfromcsv('gaia-hrd-dr3-'+str(dist)+'pc.csv')
    ra_d, dec_d = gaiarec_d.ra, gaiarec_d.dec
    bp_rp_d, photg_d = gaiarec_d.bp_rp, gaiarec_d.phot_g_mean_mag
    mg_d = photg_d + 5*np.log10(gaiarec_d.parallax) - 10  
    source_id_d = gaiarec_d.source_id
    teff_d = gaiarec_d.teff_gspphot
    # here the masked values are nans
    maskteff_d = np.where(np.isnan(teff_d))
    # for later use
    new_d = False
    # astrometric quality indicators
    astr_gof_d = gaiarec_d.astrometric_gof_al
    astr_noise_d = gaiarec_d.astrometric_excess_noise
    # non single star flag
    non_single_d = gaiarec_d.non_single_star
    # spectral line broadening parameter
    vbroad_d = gaiarec_d.vbroad
except:
	# if not saved do the query
    job = send_radial_gaia_query(dump_to_file=True, output_format="csv",
                                 output_file='gaia-hrd-dr3-'+str(dist)+'pc.csv',
                                 query_size=qsize, parallax_limit = parallax_limit)
    r_d = job.get_results()
    ra_d = r_d['ra'].data
    dec_d = r_d['dec'].data
    bp_rp_d = r_d['bp_rp'].data
    photg_d = r_d['phot_g_mean_mag'].data 
    mg_d = r_d['phot_g_mean_mag'].data + 5*np.log10(r_d['parallax'].data) - 10
    source_id_d = r_d['source_id'].data
    teff_d=r_d['teff_gspphot'].data
    # here teff is a masked array
    maskteff_d = np.where(teff_d.mask==True)
    # for later use
    new_d = True
    # astrometric quality indicators
    astr_gof_d = r_d['astrometric_gof_al'].data
    astr_noise_d = r_d['astrometric_excess_noise'].data
    # non single star flag
    non_single_d = r_d['non_single_star'].data
    # spectral line broadening parameter
    vbroad_d = r_d['vbroad'].data


# =============================================================================
# Query - magnitude limit
# =============================================================================

# set Gmag limit
gmag_lim = 9

# set query size
qsize = 500000

# check if query already saved
try:
    # if it is, read it in from file
    gaiarec_m = np.recfromcsv('gaia-hrd-dr3-'+str(gmag_lim)+'gmag.csv')
    ra_m, dec_m = gaiarec_m.ra_m, gaiarec_m.dec_m
    bp_rp_m, photg_m = gaiarec_m.bp_rp, gaiarec_m.phot_g_mean_mag
    mg_m = photg_m + 5*np.log10(gaiarec_m.parallax) - 10  
    source_id_m = gaiarec_m.source_id
    teff_m = gaiarec_m.teff_gspphot
    # here the masked values are nans
    maskteff_m = np.where(np.isnan(teff_m))
    # for later use
    new_m = False
    # astrometric quality indicators
    astr_gof_m = gaiarec_m.astrometric_gof_al
    astr_noise_m = gaiarec_m.astrometric_excess_noise
    # non single star flag
    non_single_m = gaiarec_m.non_single_star
    # spectral line broadening parameter
    vbroad_m = gaiarec_m.vbroad
except:
    # if not saved do the query
    job = send_radial_gaia_query_gmag(dump_to_file=True, output_format="csv",
                                 output_file='gaia-hrd-dr3-'+str(gmag_lim)+'gmag.csv',
                                 query_size=qsize, gmag_limit=gmag_lim)
    r_m = job.get_results()
    ra_m = r_m['ra'].data
    dec_m = r_m['dec'].data
    bp_rp_m = r_m['bp_rp'].data
    photg_m = r_m['phot_g_mean_mag'].data 
    mg_m = r_m['phot_g_mean_mag'].data + 5*np.log10(r_m['parallax'].data) - 10
    source_id_m = r_m['source_id'].data
    teff_m=r_m['teff_gspphot'].data
    # here teff is a masked array
    maskteff_m = np.where(teff_m.mask==True)
    # for later use
    new_m = True
    # astrometric quality indicators
    astr_gof_m = r_m['astrometric_gof_al'].data
    astr_noise_m = r_m['astrometric_excess_noise'].data
    # non single star flag
    non_single_m = r_m['non_single_star'].data
    # spectral line broadening parameter
    vbroad_m = r_m['vbroad'].data

# =============================================================================
# Plots
# =============================================================================

# # binned H-R diagram
# fig, ax = plt.subplots(figsize=(6, 6))
# # only show 2D-histogram for bins with more than 10 stars in them
# h = ax.hist2d(bp_rp, mg, bins=300, cmin=10, norm=colors.PowerNorm(0.5), zorder=0.5)
# # fill the rest with scatter (set rasterized=True if saving as vector graphics)
# ax.scatter(bp_rp, mg, alpha=0.05, s=1, color='k', zorder=0)
# ax.invert_yaxis()
# cb = fig.colorbar(h[3], ax=ax, pad=0.02)
# ax.set_xlabel(r'$G_{BP} - G_{RP}$')
# ax.set_ylabel(r'$M_G$')
# cb.set_label(r"$\mathrm{Stellar~density}$")

if make_plots:
    # unbinned H-R diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(bp_rp_m, mg_m, s=10, c=teff_m, zorder=0, cmap='plasma')
    ax.scatter(bp_rp_m[maskteff_m], mg_m[maskteff_m], s=3, c='k', zorder=1)
    ax.invert_yaxis()
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc)
    cb.set_label(r"$T_{eff}$")
    #fig.suptitle('H-R diagram - GAIA stars within 20 parsec')
    fig.suptitle('H-R diagram - GAIA stars with Gmag<8')
    
    # fig, ax = plt.subplots(figsize=(6, 6))
    # sc = ax.scatter(bp_rp, mg, s=6,zorder=0)
    # ax.invert_yaxis()
    # ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    # ax.set_ylabel(r'$M_G$')

# crop to G-K only
# M_G limits
M_G_G0 = 4.325
M_G_G9 = 5.34
M_G_K9 = 8.03
# colour limits
BP_RP_G0 = 0.784
BP_RP_G9 = 0.950
BP_RP_K9 = 1.79

# create masks
# G-K dwarf mask
mask_GK_m = np.where((M_G_G0 <= mg_m) &
	              (M_G_K9 >= mg_m) &
	              (BP_RP_G0 <= bp_rp_m) &
	              (BP_RP_K9 >= bp_rp_m))

# G dwarf mask
mask_G_m = np.where((M_G_G0 <= mg_m) &
                  (M_G_G9 >= mg_m) &
                  (BP_RP_G0 <= bp_rp_m) &
                  (BP_RP_G9 >= bp_rp_m))

# same for the no teff ones
if new_m:
    mask_GK_noteff_m = np.where((M_G_G0 <= mg_m) &
		              (M_G_K9 >= mg_m) &
		              (BP_RP_G0 <= bp_rp_m) &
		              (BP_RP_K9 >= bp_rp_m) &
		              (teff_m.mask==True))
    mask_G_noteff_m = np.where((M_G_G0 <= mg_m) &
                      (M_G_G9 >= mg_m) &
                      (BP_RP_G0 <= bp_rp_m) &
                      (BP_RP_G9 >= bp_rp_m) &
                      (teff_m.mask==True))
else:
    mask_GK_noteff_m = np.where((M_G_G0 <= mg_m) &
		              (M_G_K9 >= mg_m) &
		              (BP_RP_G0 <= bp_rp_m) &
		              (BP_RP_K9 >= bp_rp_m) &
		              (np.isnan(teff_m)))
    mask_G_noteff_m = np.where((M_G_G0 <= mg_m) &
                      (M_G_G9 >= mg_m) &
                      (BP_RP_G0 <= bp_rp_m) &
                      (BP_RP_G9 >= bp_rp_m) &
                      (np.isnan(teff_m)))

# masks for distance queries
# G-K dwarf mask
mask_GK_d = np.where((M_G_G0 <= mg_d) &
                  (M_G_K9 >= mg_d) &
                  (BP_RP_G0 <= bp_rp_d) &
                  (BP_RP_K9 >= bp_rp_d))

# G dwarf mask
mask_G_d = np.where((M_G_G0 <= mg_d) &
                  (M_G_G9 >= mg_d) &
                  (BP_RP_G0 <= bp_rp_d) &
                  (BP_RP_G9 >= bp_rp_d))

# same for the no teff ones
if new_d:
    mask_GK_noteff_d = np.where((M_G_G0 <= mg_d) &
                      (M_G_K9 >= mg_d) &
                      (BP_RP_G0 <= bp_rp_d) &
                      (BP_RP_K9 >= bp_rp_d) &
                      (teff_d.mask==True))
    mask_G_noteff_d = np.where((M_G_G0 <= mg_d) &
                      (M_G_G9 >= mg_d) &
                      (BP_RP_G0 <= bp_rp_d) &
                      (BP_RP_G9 >= bp_rp_d) &
                      (teff_d.mask==True))
else:
    mask_GK_noteff_d = np.where((M_G_G0 <= mg_d) &
                      (M_G_K9 >= mg_d) &
                      (BP_RP_G0 <= bp_rp_d) &
                      (BP_RP_K9 >= bp_rp_d) &
                      (np.isnan(teff_d)))
    mask_G_noteff_d = np.where((M_G_G0 <= mg_d) &
                      (M_G_G9 >= mg_d) &
                      (BP_RP_G0 <= bp_rp_d) &
                      (BP_RP_G9 >= bp_rp_d) &
                      (np.isnan(teff_d)))

if make_plots:
    # unbinned H-R diagram for GK dwarfs 
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(bp_rp_m[mask_GK_m], mg_m[mask_GK_m], s=10, c=teff_m[mask_GK_m], zorder=1, cmap='plasma')
    ax.scatter(bp_rp_m[mask_GK_noteff_m], mg_m[mask_GK_noteff_m], s=5, c='k', zorder=0)
    ax.set_xlim((0.7,1.8))
    ax.set_ylim((4,8.5))
    ax.invert_yaxis()
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc)
    cb.set_label(r"$T_{eff}$")
    
    # unbinned H-R diagram for G dwarfs only
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(bp_rp_m[mask_G_m], mg_m[mask_G_m], s=10, c=teff_m[mask_G_m], zorder=1, cmap='plasma', vmin=3900, vmax=6200)
    ax.scatter(bp_rp_m[mask_G_noteff_m], mg_m[mask_G_noteff_m], s=5, c='k', zorder=0)
    ax.set_xlim((0.7,1.))
    ax.set_ylim((4,5.5))
    ax.invert_yaxis()
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc)
    cb.set_label(r"$T_{eff}$")
    
    # do only the no-teff ones
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(bp_rp_m[mask_GK_noteff_m], mg_m[mask_GK_noteff_m], s=5, c='k', zorder=0)
    ax.set_xlim((1.8,4.9))
    ax.set_ylim((8,16.5))
    ax.invert_yaxis()
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    
    # # unbinned H-R diagram for GK dwarfs - different symbols
    # fig, ax = plt.subplots(figsize=(6, 6))
    # sc = ax.scatter(bp_rp[mask_GK], mg[mask_GK], s=10, c=teff[mask_GK], zorder=10, 
    #                 cmap='plasma', vmin=3900, vmax=6200)
    # sc = ax.scatter(bp_rp[mask_G], mg[mask_G], s=15, c=teff[mask_G], zorder=1,
    #                 cmap='plasma', vmin=3900, vmax=6200, marker='s')
    # ax.scatter(bp_rp[mask_GK_noteff], mg[mask_GK_noteff], s=5, c='k', zorder=0)
    # ax.set_xlim((0.7,1.8))
    # ax.set_ylim((4,8.5))
    # ax.invert_yaxis()
    # ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    # ax.set_ylabel(r'$M_G$')
    # cb = fig.colorbar(sc)
    # cb.set_label(r"$T_{eff}$")
    
    
    # unbinned H-R diagram for GK dwarfs and G dwarfs as subplots
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 6))
    sc = ax1.scatter(bp_rp_m[mask_GK_m], mg_m[mask_GK_m], s=10, c=teff_m[mask_GK_m], zorder=1, cmap='plasma')
    ax1.scatter(bp_rp_m[mask_GK_noteff_m], mg_m[mask_GK_noteff_m], s=5, c='k', zorder=0)
    ax1.set_xlim((0.7,1.8))
    ax1.set_ylim((4,8.5))
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('G and K dwarfs - '+str(np.shape(mask_GK_m)[1])+' stars')
    # add close but too faint stars
    lab_aux = 0
    for i in range(len(source_id_d[mask_GK_d])):
        if source_id_d[mask_GK_d][i] not in source_id_m[mask_GK_m]:
            #print(bp_rp_d[mask_GK_d][i], mg_d[mask_GK_d][i])
            if lab_aux == 0:
                ax1.scatter(bp_rp_d[mask_GK_d][i], mg_d[mask_GK_d][i], s=10, c='red', marker='X', zorder=0, label='missed')
                lab_aux = 1
            else:
                ax1.scatter(bp_rp_d[mask_GK_d][i], mg_d[mask_GK_d][i], s=10, c='red', marker='X', zorder=0)
    ax1.legend()
    #cb = fig.colorbar(sc)
    #cb.set_label(r"$T_{eff}$")
    # unbinned H-R diagram for G dwarfs only
    sc2 = ax2.scatter(bp_rp_m[mask_G_m], mg_m[mask_G_m], s=10, c=teff_m[mask_G_m], zorder=1, cmap='plasma', vmin=3900, vmax=6200)
    ax2.scatter(bp_rp_m[mask_G_noteff_m], mg_m[mask_G_noteff_m], s=5, c='k', zorder=0)
    ax2.set_xlim((0.7,1.))
    ax2.set_ylim((4,5.5))
    ax2.invert_yaxis()
    ax2.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax2.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc2)
    cb.set_label(r"$T_{eff}$")
    ax2.set_title('G dwarfs only - '+str(np.shape(mask_G_m)[1])+' stars')
    #fig.suptitle('H-R diagram - GAIA stars within 20 parsec')
    fig.suptitle('H-R diagram - GAIA stars with Gmag<8')
    
    
    # unbinned H-R diagram for GK dwarfs, missing close stars marked
    fig, ax1 = plt.subplots(figsize=(9, 6))
    sc = ax1.scatter(bp_rp_m[mask_GK_m], mg_m[mask_GK_m], s=10, c=teff_m[mask_GK_m], zorder=1, cmap='plasma')
    ax1.scatter(bp_rp_m[mask_GK_noteff_m], mg_m[mask_GK_noteff_m], s=5, c='k', zorder=0)
    ax1.set_xlim((0.7,1.8))
    ax1.set_ylim((4,8.5))
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('G and K dwarfs - '+str(np.shape(mask_GK_m)[1])+' stars')
    # add close but too faint stars
    for i in range(len(source_id_d[mask_GK_d])):
        if source_id_d[mask_GK_d][i] not in source_id_m[mask_GK_m]:
            #print(bp_rp_d[mask_GK_d][i], mg_d[mask_GK_d][i])
            ax1.scatter(bp_rp_d[mask_GK_d][i], mg_d[mask_GK_d][i], s=7, c='red', marker='X', zorder=0)
    cb = fig.colorbar(sc)
    cb.set_label(r"$T_{eff}$")
    #fig.suptitle('H-R diagram - GAIA stars within 20 parsec')
    fig.suptitle('H-R diagram - GAIA stars with Gmag<8')
    
    
    # unbinned H-R diagram for GK dwarfs and G dwarfs as subplots - distance cut
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 6))
    sc = ax1.scatter(bp_rp_d[mask_GK_d], mg_d[mask_GK_d], s=10, c=teff_d[mask_GK_d], zorder=1, cmap='plasma')
    ax1.scatter(bp_rp_d[mask_GK_noteff_d], mg_d[mask_GK_noteff_d], s=5, c='k', zorder=0)
    ax1.set_xlim((0.7,1.8))
    ax1.set_ylim((4,8.5))
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('G and K dwarfs - '+str(np.shape(mask_GK_d)[1])+' stars')
    #cb = fig.colorbar(sc)
    #cb.set_label(r"$T_{eff}$")
    # unbinned H-R diagram for G dwarfs only
    sc2 = ax2.scatter(bp_rp_d[mask_G_d], mg_d[mask_G_d], s=10, c=teff_d[mask_G_d], zorder=1, cmap='plasma', vmin=3900, vmax=6200)
    ax2.scatter(bp_rp_d[mask_G_noteff_d], mg_d[mask_G_noteff_d], s=5, c='k', zorder=0)
    ax2.set_xlim((0.7,1.))
    ax2.set_ylim((4,5.5))
    ax2.invert_yaxis()
    ax2.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax2.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc2)
    cb.set_label(r"$T_{eff}$")
    ax2.set_title('G dwarfs only - '+str(np.shape(mask_G_d)[1])+' stars')
    fig.suptitle('H-R diagram - GAIA stars within 20 parsec')

# select which to use
if query_type == 'dist':
    #reset all values
    ra = ra_d
    dec = dec_d
    bp_rp = bp_rp_d
    photg = photg_d
    mg = mg_d
    source_id = source_id_d
    teff=teff_d
    maskteff = maskteff_d
    astr_gof = astr_gof_d
    astr_noise = astr_noise_d
    non_single = non_single_d
    vbroad = vbroad_d
    mask_GK = mask_GK_d
    mask_G = mask_G_d
    mask_GK_noteff = mask_GK_noteff_d
    mask_G_noteff = mask_G_noteff_d
elif query_type == 'mag':
    #reset all values
    ra = ra_m
    dec = dec_m
    bp_rp = bp_rp_m
    photg = photg_m
    mg = mg_m
    source_id = source_id_m
    teff=teff_m
    maskteff = maskteff_m
    astr_gof = astr_gof_m
    astr_noise = astr_noise_m
    non_single = non_single_m
    vbroad = vbroad_m
    mask_GK = mask_GK_m
    mask_G = mask_G_m
    mask_GK_noteff = mask_GK_noteff_m
    mask_G_noteff = mask_G_noteff_m



# =============================================================================
# Check against exoplanet archive
# =============================================================================

# read the archive table
# preferred parameters only
# delete the reference columns, they break the import somehow
exop = np.recfromcsv('/home/hobson/Documents/Second_Earth_initiative/exoplanets_no_ref_cols.csv', skip_header=101)

# get the gaia IDs
# need to strip leading characters and decode
gaia_exop_1 = np.char.strip(exop['gaia_id'], b'Gaia DR2 ')
gaia_exop_2 = np.array([x.decode() for x in gaia_exop_1])

# mask systems with no gaia ID 
gaia_id_mask = np.where(gaia_exop_2!='')
gaia_exop_3 = gaia_exop_2[gaia_id_mask]

# convert to float
gaia_exop_4 = np.array([x.astype(float) for x in gaia_exop_3])

# find and store cross-matches
stars_with_planets_exop = []
stars_with_planets_gaia = []
stars_with_planets_gstars = []

#also store number of planets
number_of_planets = []

# loop over source IDs - GK stars
for i in range(np.shape(mask_GK)[1]):
    # get the ID
    test_id_dr3 = source_id[mask_GK][i]
    # need the DR2 ID! Query it from Simbad:
    try:
        simbad_ids = Simbad.query_objectids("Gaia DR3 "+str(test_id_dr3))
        for j in range(len(simbad_ids)):
                simb_id = (simbad_ids[j][0]).replace(' ','')
                if 'GaiaDR2' in simb_id:
                    test_id = int(simb_id.strip('GaiaDR2'))
    except:
        # dummy value if it can't be found
        test_id = 999
    # if it's in the exoplanet archive
    if test_id in gaia_exop_4:
        # get both rec numbers and store them
        rec = np.where(gaia_exop_4==test_id)
        stars_with_planets_exop.append(rec[0])
        # as one system may have multiple planets, store i repeatedly
        stars_with_planets_gaia.append(np.repeat(i, len(rec[0])))
        # also check if it's a G str and store the ID if it is
        if len(np.where(source_id[mask_G]==test_id)[0]) > 0:
            stars_with_planets_gstars.append(np.where(source_id[mask_G]==test_id)[0])
        # save number of planets
        number_of_planets.append(len(rec[0]))
    # if not note 0 planets
    number_of_planets.append(0)

# flatten the lists
stars_with_planets_exop_f = np.concatenate(stars_with_planets_exop).ravel()
stars_with_planets_gaia_f = np.concatenate(stars_with_planets_gaia).ravel()
stars_with_planets_gstars_f = np.concatenate(stars_with_planets_gstars).ravel()

if make_plots:

    # mass-period diagram
    # only one without mass, HD 110082 b; only 9/52 with radii
    # colour by star mass
    # error arrays
    orbper_error = np.vstack((exop['pl_orbpererr1'][stars_with_planets_exop_f],
                              exop['pl_orbpererr2'][stars_with_planets_exop_f]))
    massj_error = np.vstack((exop['pl_bmassjerr1'][stars_with_planets_exop_f],
                             exop['pl_bmassjerr2'][stars_with_planets_exop_f]))
    # plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(exop['pl_orbper'][stars_with_planets_exop_f], exop['pl_bmassj'][stars_with_planets_exop_f],
                c=teff[mask_GK][stars_with_planets_gaia_f].data, cmap='plasma', vmin=3900, vmax=6200)
    #ax.errorbar(exop['pl_orbper'][stars_with_planets_exop_f], exop['pl_bmassj'][stars_with_planets_exop_f],
    #            xerr=orbper_error, yerr=massj_error)
    ax.set_xlabel('Orbital period [d]')
    ax.set_ylabel('Planet mass [MJ]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    cb=plt.colorbar(sc)
    cb.set_label(r"$T_{eff}$")
    
    # # plot systems for each star
    # # labels
    # ylabels = np.unique(exop['hostname'][stars_with_planets_exop_f], return_index=True)[0]
    # yticks = np.unique(exop['hostname'][stars_with_planets_exop_f], return_index=True)[1]
    # #plot
    # fig = plt.figure()
    # plt.scatter(exop['pl_orbper'][stars_with_planets_exop_f], teff[mask_GK][stars_with_planets_gaia_f].data)
    # #ax.set_yticks = teff[mask_GK][stars_with_planets_gaia_f].data[yticks].tolist()
    # #ax.set_yticklabels = ylabels.tolist()
    # plt.xlabel('Orbital period [d]')
    # #ax.set_ylabel('Planet mass [MJ]')
    # plt.xscale('log')
    # #ax.set_yscale('log')
    # for i in range(len(ylabels)):
    #     plt.axhline(teff[mask_GK][stars_with_planets_gaia_f].data[yticks].tolist()[i])
    #     if i%2:
    #         plt.text(100000,teff[mask_GK][stars_with_planets_gaia_f].data[yticks].tolist()[i],  
    #                  ylabels[i].decode('utf-8'))
    #     else:
    #                 plt.text(110000,teff[mask_GK][stars_with_planets_gaia_f].data[yticks].tolist()[i],  
    #                  ylabels[i].decode('utf-8'))
    # plt.xlim(0,110000)# 
    
    # unbinned H-R diagram for GK dwarfs and G dwarfs as subplots - stars with planets highlighted
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 6))
    sc = ax1.scatter(bp_rp[mask_GK], mg[mask_GK], s=10, c=teff[mask_GK], zorder=1, cmap='plasma', vmin=3900, vmax=6200)
    ax1.scatter(bp_rp[mask_GK][np.unique(stars_with_planets_gaia_f)], mg[mask_GK][np.unique(stars_with_planets_gaia_f)],
                s=20, c=teff[mask_GK][np.unique(stars_with_planets_gaia_f)], zorder=10, cmap='plasma', vmin=3900, vmax=6200, edgecolors='k')
    #ax1.scatter(bp_rp[mask_GK_noteff], mg[mask_GK_noteff], s=5, c='k', zorder=0)
    ax1.set_xlim((0.7,1.8))
    ax1.set_ylim((4,8.5))
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax1.set_ylabel(r'$M_G$')
    ax1.set_title('G and K dwarfs - '+str(np.shape(mask_GK)[1])+' stars - '+str(len(np.unique(stars_with_planets_gaia_f)))+' planet hosts')
    #cb = fig.colorbar(sc)
    #cb.set_label(r"$T_{eff}$")
    # unbinned H-R diagram for G dwarfs only
    sc2 = ax2.scatter(bp_rp[mask_G], mg[mask_G], s=10, c=teff[mask_G], zorder=1, cmap='plasma', vmin=3900, vmax=6200)
    ax2.scatter(bp_rp[mask_G][np.unique(stars_with_planets_gstars_f)], mg[mask_G][np.unique(stars_with_planets_gstars_f)], 
                s=20, c=teff[mask_G][np.unique(stars_with_planets_gstars_f)], zorder=10, cmap='plasma', vmin=3900, vmax=6200, edgecolors='k')
    #ax2.scatter(bp_rp[mask_G_noteff], mg[mask_G_noteff], s=5, c='k', zorder=0)
    ax2.set_xlim((0.7,1.))
    ax2.set_ylim((4,5.5))
    ax2.invert_yaxis()
    ax2.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax2.set_ylabel(r'$M_G$')
    cb = fig.colorbar(sc2)
    cb.set_label(r"$T_{eff}$")
    ax2.set_title('G dwarfs only - '+str(np.shape(mask_G)[1])+' stars - '+str(len(np.unique(stars_with_planets_gstars_f)))+' planet hosts')
    fig.suptitle('H-R diagram - GAIA stars within 20 parsec - planet hosts highlighted')

# =============================================================================
# Query ESO archive
# =============================================================================

# initialize obs count
obs_count_HARPS_GK = []
obs_count_HARPS_G = []
obs_count_FEROS_GK = []
obs_count_FEROS_G = []
obs_count_ESPRESSO_GK = []
obs_count_ESPRESSO_G = []

# observability mask 
obsm_GK = np.where((dec[mask_GK]<30) & (dec[mask_GK]>-120))
obsm_G = np.where((dec[mask_G]<30) & (dec[mask_G]>-120))

# loop over the stars
for i in range(np.shape(source_id[mask_GK][obsm_GK])[0]):
    # get the ID
    test_id = source_id[mask_GK][obsm_GK][i]
    query_id = 'GAIA DR3 '+str(test_id)
    print('querying archive for '+query_id)
    # do the query
    dataH=eso.query_main(column_filters={'instrument':'HARPS','target':query_id},
                        columns=('OBJECT','RA','DEC','Program_ID','Instrument',
                                 'Category','Type','Mode','Dataset ID','Release_Date',
                                 'TPL ID','TPL START','Exptime','Exposure',
                                 'filter_lambda_min','filter_lambda_max','MJD-OBS'))
    dataF=eso.query_main(column_filters={'instrument':'FEROS','target':query_id},
                        columns=('OBJECT','RA','DEC','Program_ID','Instrument',
                                 'Category','Type','Mode','Dataset ID','Release_Date',
                                 'TPL ID','TPL START','Exptime','Exposure',
                                 'filter_lambda_min','filter_lambda_max','MJD-OBS'))
    dataE=eso.query_main(column_filters={'instrument':'ESPRESSO','target':query_id},
                        columns=('OBJECT','RA','DEC','Program_ID','Instrument',
                                 'Category','Type','Mode','Dataset ID','Release_Date',
                                 'TPL ID','TPL START','Exptime','Exposure',
                                 'filter_lambda_min','filter_lambda_max','MJD-OBS'))
    # store data length
    # check if there's data, if not append 0
    try:
        obs_count_HARPS_GK.append(len(dataH))
    except:
        obs_count_HARPS_GK.append(0)
    try:
        obs_count_FEROS_GK.append(len(dataF))
    except:
        obs_count_FEROS_GK.append(0)
    try:
        obs_count_ESPRESSO_GK.append(len(dataE))
    except:
        obs_count_ESPRESSO_GK.append(0)
    # also check if it's a G str and store the data length if it is
    if len(np.where(source_id[mask_G]==test_id)[0]) > 0:
        try:
            obs_count_HARPS_G.append(len(dataH))
        except:
            obs_count_HARPS_G.append(0)
        try:
            obs_count_FEROS_G.append(len(dataF))
        except:
            obs_count_FEROS_G.append(0)
        try:
            obs_count_ESPRESSO_G.append(len(dataE))
        except:
            obs_count_ESPRESSO_G.append(0)

# convert to arrays
obs_count_HARPS_GK = np.array(obs_count_HARPS_GK)
obs_count_HARPS_G = np.array(obs_count_HARPS_G)
obs_count_FEROS_GK = np.array(obs_count_FEROS_GK)
obs_count_FEROS_G = np.array(obs_count_FEROS_G)
obs_count_ESPRESSO_GK = np.array(obs_count_ESPRESSO_GK)
obs_count_ESPRESSO_G = np.array(obs_count_ESPRESSO_G)

if make_plots:
    # histograms
    fig, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(12, 6))
    ax1.hist(obs_count_FEROS_GK,np.arange(0,1300,100))
    ax1.set_title('FEROS observations')
    ax2.hist(obs_count_HARPS_GK,np.arange(0,13400,100))
    ax2.set_title('HARPS observations')
    ax3.hist(obs_count_ESPRESSO_GK,np.arange(0,1000,100))
    ax3.set_title('ESPRESSO observations')
    
    # remove the zeroes
    fig, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(12, 6))
    ax1.hist(obs_count_FEROS_GK,np.arange(1,1301,50))
    ax1.set_title('FEROS observations')
    ax2.hist(obs_count_HARPS_GK,np.arange(1,13401,50))
    ax2.set_title('HARPS observations')
    ax3.hist(obs_count_ESPRESSO_GK,np.arange(1,1001,50))
    ax3.set_title('ESPRESSO observations')

# get those that are observed at least once 

GK_w_obs = np.where((obs_count_ESPRESSO_GK +
                     obs_count_FEROS_GK +
                     obs_count_HARPS_GK) >0) 

names_GK = source_id[mask_GK][obsm_GK][GK_w_obs].astype(str)

G_w_obs = np.where((obs_count_ESPRESSO_G +
                     obs_count_FEROS_G +
                     obs_count_HARPS_G) >0) 

names_G = source_id[mask_G][obsm_G][G_w_obs].astype(str)

if make_plots:
    # bar plots
    plt.figure()
    plt.bar(names_GK, obs_count_FEROS_GK[GK_w_obs],
            label='FEROS')
    plt.bar(names_GK, obs_count_HARPS_GK[GK_w_obs], 
            bottom = obs_count_FEROS_GK[GK_w_obs],
             label = 'HARPS')
    plt.bar(names_GK, obs_count_ESPRESSO_GK[GK_w_obs],
            bottom = obs_count_FEROS_GK[GK_w_obs] + obs_count_HARPS_GK[GK_w_obs],
             label = 'ESPRESSO')
    plt.title('RV observations - '+str(len(GK_w_obs[0])) +'/'+str(len(obs_count_FEROS_GK))+ 'G and K stars')
    plt.legend()
    plt.tight_layout()
    
    # bar plots
    plt.figure()
    plt.bar(names_G, obs_count_FEROS_G[G_w_obs],
            label='FEROS')
    plt.bar(names_G, obs_count_HARPS_G[G_w_obs], 
            bottom = obs_count_FEROS_G[G_w_obs],
             label = 'HARPS')
    plt.bar(names_G, obs_count_ESPRESSO_G[G_w_obs],
            bottom = obs_count_FEROS_G[G_w_obs] + obs_count_HARPS_G[G_w_obs],
             label = 'ESPRESSO')
    plt.title('RV observations - '+str(len(G_w_obs[0])) +'/'+str(len(obs_count_FEROS_G))+ 'G stars')
    plt.legend()
    plt.tight_layout()

# save file with names, coordinates, observation count
# GK stars
save_array_GK = np.column_stack((source_id[mask_GK][obsm_GK][GK_w_obs], obs_count_FEROS_GK[GK_w_obs],obs_count_HARPS_GK[GK_w_obs],obs_count_ESPRESSO_GK[GK_w_obs]))
np.savetxt('obs_counts_GK.txt', save_array_GK, fmt='%d', delimiter='\t', header='GAIA_ID\tFEROS_obs\tHARPS_obs\tESPRESSO_obs')
# G stars only
save_array_G = np.column_stack((source_id[mask_G][obsm_G][G_w_obs], obs_count_FEROS_G[G_w_obs],obs_count_HARPS_G[G_w_obs],obs_count_ESPRESSO_G[G_w_obs]))
np.savetxt('obs_counts_G.txt', save_array_G,fmt='%d', delimiter='\t', header='GAIA_ID\tFEROS_obs\tHARPS_obs\tESPRESSO_obs')

# =============================================================================
# Query RVbank
# =============================================================================

# RV bank URL
RVBank_url = 'https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank_v1.csv'

# get date 
today = datetime.today().strftime('%Y-%m-%d')

# check if there's an old file
RVBank_old = glob.glob('HARPS_RVBank_*.csv')
if len(RVBank_old) == 1:
    print('old RV bank exists from '+RVBank_old[0][13:23])
elif len(RVBank_old) == 0:
    print('no old RV bank exists')

# check if bank is from today
if RVBank_old[0][13:23] == today:
    print('RV bank already updated today, skipping download')
else:
    print('Retrieving new RV bank')
    # get the RV bank and name it with today's date
    try:
        wget.download(RVBank_url, 'HARPS_RVBank_'+today+'.csv')
        print('\nretrieval successful, removing old RV bank if existent')
        if len(RVBank_old) == 1:
            os.remove(RVBank_old[0])
    # if the retrieval fails, use the old one or kill the process
    except: 
        if len(RVBank_old) == 1:
            print('retrieval failed, using previous RV bank from '+RVBank_old[0][13:23])
        elif len(RVBank_old) == 0:
            print('retrieval failed and no previous RV bank stored, cannot proceed')
            sys.exit()

# get the name of the new RV bank
RVBank_file = glob.glob('HARPS_RVBank_*.csv')
print('reading in RV bank '+RVBank_file[0])
RVBank = np.genfromtxt(RVBank_file[0], delimiter=',', usecols=np.arange(0,51), 
                       names=True, dtype=None)

# aux targets array decoded 
targets = np.char.decode(RVBank['target'])

# store matching IDs
match_ID_gaia_G = []
match_ID_simbad_G = []
RV_RMS = []
RV_med_err = []
FWHM_med = []
number_matches = []

# loop over all stars
for i in range(len(source_id[mask_GK][obsm_GK])):
#for i in range(3):
    if obs_count_HARPS_GK[i]>0:
        print('seeking match for Gaia DR3 '+str(source_id[mask_GK][obsm_GK][i]))
        # get all names from simbad
        try:
            simbad_ids = Simbad.query_objectids("Gaia DR3 "+str(source_id[mask_GK][obsm_GK][i]))
            #print(simbad_ids)
            found = 0
            for j in range(len(simbad_ids)):
                simb_id = (simbad_ids[j][0]).replace(' ','')
                if simb_id in targets:
                    found +=1
                    match_ID_gaia_G.append(str(source_id[mask_GK][obsm_GK][i]))
                    match_ID_simbad_G.append(simb_id)
                    RV_index = np.where(targets == simb_id)
        except:
            found = -1
            # store RVBankID as None
            match_ID_simbad_G.append('NoSimbad')
        # print match or no match
        if found == 1:
            print('match found under ID '+match_ID_simbad_G[-1])
        elif found == 0:
            print('no match found!')
            # store RVBankID as None
            match_ID_simbad_G.append('NoRVBank')
        elif found>1:
            print('multiple matches found!')
            # store RVBankID as None
            match_ID_simbad_G.append('multiRVBank')

        #store number of matches
        number_matches.append(found)
        # if there is a match, use these RVs
        if found == 1:
            # get data for this star
            star = RVBank[RV_index]
            # get number of data points
            obs_total = len(star)
            # get first and last date
            t = Time(np.array((star['BJD'][0],star['BJD'][-1])), format='jd')

            # mask RV outliers
            RV_sig = np.std(star['RV_mlc_nzp'])
            RV_median = np.median(star['RV_mlc_nzp'])
            mask = np.where(np.abs(star['RV_mlc_nzp']-RV_median)<=3*RV_sig)

            # median over nights

            tt=Time(star['BJD'][mask], format='jd')
            j = 0
            BJD_nmed = []
            RV_nmed = []
            CRX_nmed = []
            Ha_nmed = []
            Na1_nmed = []
            Na2_nmed = []
            e_RV_nmed = []
            e_CRX_nmed = []
            e_Ha_nmed = []
            e_Na1_nmed = []
            e_Na2_nmed = []
            BIS_nmed = []
            e_BIS_nmed = []
            FWHM_nmed = []
            e_FWHM_nmed = []

                
            while j<len(tt.value):
                mm = np.where(np.floor(tt.value) == np.floor(tt.value[j]))
                BJD_nmed.append(np.median(tt.value[mm]))
                RV_nmed.append(np.median(star['RV_mlc_nzp'][mask][mm]))
                CRX_nmed.append(np.median(star['CRX'][mask][mm]))
                Ha_nmed.append(np.median(star['Halpha'][mask][mm]))
                Na1_nmed.append(np.median(star['NaD1'][mask][mm]))
                Na2_nmed.append(np.median(star['NaD2'][mask][mm]))
                e_RV_nmed.append(np.median(star['e_RV_mlc_nzp'][mask][mm]))
                e_CRX_nmed.append(np.median(star['e_CRX'][mask][mm]))
                e_Ha_nmed.append(np.median(star['e_Halpha'][mask][mm]))
                e_Na1_nmed.append(np.median(star['e_NaD1'][mask][mm]))
                e_Na2_nmed.append(np.median(star['e_NaD2'][mask][mm]))
                BIS_nmed.append(np.median(star['BIS'][mask][mm]))
                FWHM_nmed.append(np.median(star['FWHM_DRS'][mask][mm]))
                j = mm[0][-1]+1
            # print info
            print(str(len(tt.value))+' observations over '+str(len(BJD_nmed))+' nights')

            # get RV RMS
            rv_rms = np.sqrt(np.mean(np.square(RV_nmed)))
            RV_RMS.append(rv_rms)

            # get median FWHM
            FWHM_med.append(np.median(FWHM_nmed))

            # get median RV uncertainty
            RV_med_err.append(np.median(e_RV_nmed))

        # if no match
        else:
            RV_RMS.append(np.nan)
            FWHM_med.append(np.nan)
            RV_med_err.append(np.nan)
    # if no HARPS obs
    else:
        RV_RMS.append(np.nan)
        FWHM_med.append(np.nan)
        RV_med_err.append(np.nan)
        # store RVBankID as None
        match_ID_simbad_G.append('NoHARPS')

# =============================================================================
# Query Simbad for IDs and Vmags
# =============================================================================

# get names in proper formatting
names_gdr3 = ["Gaia DR3 "+ str(gdr3id) for gdr3id in source_id[mask_GK][obsm_GK]]

# add Vmag, Jmag, rot, spectral type to Simbad fields queried
Simbad.add_votable_fields('flux(V)')
Simbad.add_votable_fields('flux(J)')
Simbad.add_votable_fields('rot')
Simbad.add_votable_fields('sp')

#initialize
Simbad_names = []
Simbad_Vmags = []
Simbad_Jmags = []
Simbad_Vsini = []
Simbad_sp = []

# do it on a loop so as to get the ones that fail too
for name in names_gdr3:
    try:
        test = Simbad.query_object(name)
        Simbad_names.append(test['MAIN_ID'][0])
        Simbad_Vmags.append(test['FLUX_V'][0])
        Simbad_Jmags.append(test['FLUX_J'][0])
        Simbad_Vsini.append(test['ROT_Vsini'][0])
        Simbad_sp.append(test['SP_TYPE'][0])

    except:
        Simbad_names.append('NoSimbad')
        Simbad_Vmags.append(np.nan)
        Simbad_Jmags.append(np.nan)
        Simbad_Vsini.append(np.nan)
        Simbad_sp.append('NoSP')

# set the "masked" values to 999 so I can write the columns properly
# for i in range(len(Simbad_names)):
#     if Simbad_Vsini[i] == 'masked':
#         Simbad_Vsini[i] = 999
#     if Simbad_Vmags[i] == 'masked':
#         Simbad_Vmags[i] = 999

# =============================================================================
# Cross-match with Ansgar's RV precision table
# =============================================================================

# read the table
RV_prec = np.genfromtxt('/home/hobson/Documents/Second_Earth_initiative/Ansgar Reiners/1614175056075_targets2ndEarth_cleandupes.csv',
                        delimiter=',', dtype=None, names=True)

# get the names
RV_prec_names = np.char.decode(RV_prec['ID'])
# strip whitespace
RV_prec_names_s = [x.replace(' ','') for x in RV_prec_names]
RV_prec_names_s = np.array(RV_prec_names_s)

#initialize storage
RV_prec_table = []
HZ_mass_table = []
matches = []

# loop over the stars and seek matches in 2MASS/HD/HIP names
for i in range(len(source_id[mask_GK][obsm_GK])):
    print('Getting RV precision match for '+str(source_id[mask_GK][obsm_GK][i]))
    try:
        # get all IDs from Simbad:
        simbad_ids = Simbad.query_objectids("Gaia DR3 "+str(source_id[mask_GK][obsm_GK][i]))
        # match checker
        match = 0
        # try to match on 2MASS
        for j in range(len(simbad_ids)):
            # find 2 MASS ID
            if '2MASS' in simbad_ids[j][0].replace(' ',''):
                # check for match
                if simbad_ids[j][0].replace(' ','') in RV_prec_names_s:
                    # get the index of match
                    ind = np.argwhere(RV_prec_names_s == simbad_ids[j][0].replace(' ',''))
                    # store the RV precision and HZ mass
                    RV_prec_table.append(RV_prec['2ndEarth'][ind[0]][0])
                    HZ_mass_table.append(RV_prec['2ndEarth_1'][ind[0]][0])
                    match += 1
                    print('2MASS match')
            # find HD ID
            elif 'HD' in simbad_ids[j][0].replace(' ',''):
                # check for match
                if simbad_ids[j][0].replace(' ','') in RV_prec_names_s:
                    # get the index of match
                    ind = np.argwhere(RV_prec_names_s == simbad_ids[j][0].replace(' ',''))
                    # store the RV precision and HZ mass
                    RV_prec_table.append(RV_prec['2ndEarth'][ind[0]][0])
                    HZ_mass_table.append(RV_prec['2ndEarth_1'][ind[0]][0])
                    match += 1
                    print('HD match')
            # find HIP ID
            elif 'HIP' in simbad_ids[j][0].replace(' ',''):
                # check for match
                if simbad_ids[j][0].replace(' ','') in RV_prec_names_s:
                    # get the index of match
                    ind = np.argwhere(RV_prec_names_s == simbad_ids[j][0].replace(' ',''))
                    # store the RV precision and HZ mass
                    RV_prec_table.append(RV_prec['2ndEarth'][ind[0]][0])
                    HZ_mass_table.append(RV_prec['2ndEarth_1'][ind[0]][0])
                    match += 1
                    print('HIP match')
        # if no match save NaNs
        if match == 0:
            RV_prec_table.append(np.nan)
            HZ_mass_table.append(np.nan)
            print('no match')
    except:
        RV_prec_table.append(np.nan)
        HZ_mass_table.append(np.nan)
        print('no Simbad')
    matches.append(match)

# =============================================================================
# Save table
# =============================================================================


# save file with names, coordinates, observation counts, RV RMS, FWHM, RV err, number of planets
# GK stars
#save_array_GK = np.column_stack((source_id[mask_GK][obsm_GK], Simbad_names, match_ID_simbad_G, Simbad_Vmags, Simbad_Vsini,
#                                 obs_count_FEROS_GK,obs_count_HARPS_GK,obs_count_ESPRESSO_GK,
#                                RV_RMS, FWHM_med, RV_med_err, np.array(number_of_planets)[obsm_GK]))
#
#save_array_GK = np.array(zip(source_id[mask_GK][obsm_GK], Simbad_names, match_ID_simbad_G, Simbad_Vmags, Simbad_Vsini,
#                                 obs_count_FEROS_GK,obs_count_HARPS_GK,obs_count_ESPRESSO_GK,
#                                RV_RMS, FWHM_med, RV_med_err, np.array(number_of_planets)[obsm_GK]),
#                            dtype=[('source_id[mask_GK][obsm_GK]',int), Simbad_names, match_ID_simbad_G, Simbad_Vmags, Simbad_Vsini,
#                                 obs_count_FEROS_GK,obs_count_HARPS_GK,obs_count_ESPRESSO_GK,
#                                RV_RMS, FWHM_med, RV_med_err, np.array(number_of_planets)[obsm_GK]])
#
#save_array_GK = np.array(zip(source_id[mask_GK][obsm_GK], Simbad_names),
#                            dtype=[('t1',int), ('t2', 'S16')])

save_array_GK_dtypes = np.dtype([('GDR3id',int),('Simbad_names', 'U16'), ('RVBank_ID','U16'), ('Simbad_Vmags',float), ('Simbad_Jmags',float), 
                                 ('Gmags', float), ('Simbad_sp','U16'),('Simbad_Vsini',float),
                                 ('obs_FEROS', int),('obs_HARPS',int),('obs_ESPRESSO', int),
                                 ('RV_RMS',float), ('FWHM_med',float), ('RV_med_err', float), ('number_of_planets', int), ('RV_prec_limit', float), ('HZ_mass_limit', float)])
save_array_GK = np.empty(len(source_id[mask_GK][obsm_GK]),dtype=save_array_GK_dtypes)
save_array_GK['GDR3id']=source_id[mask_GK][obsm_GK]
save_array_GK['Simbad_names']=Simbad_names
save_array_GK['RVBank_ID'] = match_ID_simbad_G
save_array_GK['Simbad_Vmags'] = Simbad_Vmags
save_array_GK['Simbad_Jmags'] = Simbad_Jmags
save_array_GK['Gmags'] = photg[mask_GK][obsm_GK]
save_array_GK['Simbad_sp'] = Simbad_sp
save_array_GK['Simbad_Vsini'] = Simbad_Vsini
save_array_GK['obs_FEROS'] = obs_count_FEROS_GK
save_array_GK['obs_HARPS'] = obs_count_HARPS_GK
save_array_GK['obs_ESPRESSO'] = obs_count_ESPRESSO_GK
save_array_GK['RV_RMS'] = RV_RMS
save_array_GK['FWHM_med'] = FWHM_med
save_array_GK['RV_med_err'] = RV_med_err
save_array_GK['number_of_planets'] = np.array(number_of_planets)[obsm_GK]
save_array_GK['RV_prec_limit'] = np.array(RV_prec_table)
save_array_GK['HZ_mass_limit'] = np.array(HZ_mass_table)

np.savetxt('table_GK.txt', save_array_GK, fmt='%d\t%s\t%s\t%.2f\t%.2f\t%.2f\t%s\t%.2f\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%.2f\t%.2f', 
    header='GAIA_ID\tSimbad_ID\tRVBank_ID\tVmag\tJmag\tGmag\tsp_type\tVsini_simbad\tFEROS_obs'+
           '\tHARPS_obs\tESPRESSO_obs\tRV_RMS\tFWHM\tRV_err\tPlanets\tRV_prec_lim\tHZ_mass_lim')


# save file with IDs of stars with Vmag and SPtype - RV precision calculator input

#Vmag mask
Vmag_mask = np.where(~np.isnan(Simbad_Vmags))

#Sp type mask
spmask = np.where((tt=='') | (tt=='NoSP'))

Vmag_SP_mask = Vmag_mask[0]
del_ind = []
# remove SP masked elements 
for i in range(len(Vmag_SP_mask)):
    if Vmag_SP_mask[i] in spmask[0]:
        del_ind.append(i)

Vmag_SP_mask = np.delete(Vmag_SP_mask, del_ind)

# names array masked
names_RVcalc = np.array(names_gdr3)[Vmag_SP_mask]

np.savetxt('RV_calc_input.txt', names_RVcalc, fmt='%s')