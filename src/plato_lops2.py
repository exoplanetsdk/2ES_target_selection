import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
import astropy.units as u
import pandas as pd
import numpy as np
from config import GAIA_FILE, DETECTION_LIMITS, FIGURES_DIRECTORY

# Define LOPS2 center coordinates (ICRS J2000)
ra_center_deg = 95.310417  # Right Ascension in degrees
dec_center = "-47d53m13s"  # Declination in dms

# Convert to SkyCoord object
lops2_center = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center, frame='icrs')

# Define the field of view (49° x 49°)
fov_size = 49 * u.deg

# Create a WCS object for plotting in equatorial coordinates
wcs = WCS(naxis=2)
wcs.wcs.crval = [lops2_center.ra.deg, lops2_center.dec.deg]  # Center of the field
wcs.wcs.crpix = [0.5, 0.5]  # Reference pixel
wcs.wcs.cdelt = [-1, 1]  # Pixel scale (deg/pixel)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Projection type

# Read GAIA stars from the Excel file
gaia_data = pd.read_excel(GAIA_FILE, dtype={'source_id': str, 'source_id_dr2': str, 'source_id_dr3': str}, header=0)

# Ensure RA and DEC columns are numeric and clean any invalid data
gaia_data['RA'] = pd.to_numeric(gaia_data['RA'], errors='coerce')
gaia_data['DEC'] = pd.to_numeric(gaia_data['DEC'], errors='coerce')
gaia_data = gaia_data.dropna(subset=['RA', 'DEC'])  # Drop rows with NaN in RA or DEC

# Clean column names to make them valid Python identifiers
gaia_data.columns = gaia_data.columns.str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)

# Convert the full GAIA dataset to SkyCoord
gaia_coords = SkyCoord(ra=gaia_data['RA'].values * u.deg, dec=gaia_data['DEC'].values * u.deg, frame='icrs')

# Iterate over detection limits and plot
for detection_limit in DETECTION_LIMITS:
    # Filter stars based on detection limit (if None, include all stars)
    if detection_limit is not None:
        filtered_stars = gaia_data[gaia_data['HZ_Detection_Limit_M_Earth'] <= detection_limit]
        # Ensure RA and DEC columns are valid after filtering
        filtered_stars = filtered_stars.dropna(subset=['RA', 'DEC'])
        filtered_coords = SkyCoord(ra=filtered_stars['RA'].values * u.deg, 
                                   dec=filtered_stars['DEC'].values * u.deg, 
                                   frame='icrs')
    else:
        filtered_coords = gaia_coords
        filtered_stars = gaia_data

    # Plot the field
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_title(f"PLATO LOPS2 Field (Detection Limit: {detection_limit} M_Earth)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Right Ascension (J2000)", fontsize=14)
    ax.set_ylabel("Declination (J2000)", fontsize=14)

    # Add a square representing the FoV
    fov_square = SphericalCircle((lops2_center.ra, lops2_center.dec),
                                 radius=fov_size / 2,
                                 edgecolor='blue', facecolor='none', linestyle='-', linewidth=2,
                                 transform=ax.get_transform('icrs'))
    ax.add_patch(fov_square)

    # Add a red cross at the center
    ax.plot(lops2_center.ra.deg, lops2_center.dec.deg, 'rx', markersize=12, markeredgewidth=3,
            transform=ax.get_transform('icrs'), label='LOPS2 Center')

    # Overplot stars
    ax.plot(filtered_coords.ra.deg, filtered_coords.dec.deg, 'ko', markersize=3,
            transform=ax.get_transform('icrs'), label='GAIA Stars', alpha=0.5)

    # Label stars with detection limit below 3 and inside the circle
    if detection_limit is not None and detection_limit < 3:
        for star in filtered_stars.itertuples():
            # Convert star coordinates to SkyCoord
            star_coord = SkyCoord(ra=star.RA * u.deg, dec=star.DEC * u.deg, frame='icrs')
            # Calculate separation from the center
            separation = star_coord.separation(lops2_center)
            
            # Check if the star is within the field of view
            if separation <= fov_size / 2:
                # Try to retrieve the label from HD, GJ, or HIP columns
                label = None
                if pd.notna(star.HD_Number) and str(star.HD_Number).strip():
                    label = f"{str(star.HD_Number).strip().replace(' ', '')}"
                elif pd.notna(star.GJ_Number) and str(star.GJ_Number).strip():
                    label = f"{str(star.GJ_Number).strip().replace(' ', '')}"
                elif pd.notna(star.HIP_Number) and str(star.HIP_Number).strip():
                    label = f"{str(star.HIP_Number).strip().replace(' ', '')}"

                # If a label is found, add it to the plot
                if label:
                    ax.text(star.RA + 0.01, star.DEC + 0.01, label, fontsize=14, color='red', 
                            transform=ax.get_transform('icrs'), ha='left', va='bottom')

    # Set gridlines and limits
    ax.grid(color='gray', linestyle='--', alpha=0.7)
    ax.set_xlim(-fov_size.value / 2 - 5, fov_size.value / 2 + 5)
    ax.set_ylim(-fov_size.value / 2 - 5, fov_size.value / 2 + 5)

    # Modify the RA axis to display values in degrees
    ra_axis = ax.coords[0]  # RA axis
    ra_axis.set_format_unit(u.deg)  # Set RA format to degrees
    ra_axis.set_major_formatter('d')  # Use integer degrees for tick labels

    # Add a legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # Save or show the plot
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}PLATO_LOPS2_DetectionLimit_{detection_limit}.png")
    plt.close()
