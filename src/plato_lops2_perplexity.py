import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
import astropy.units as u

# Define LOPS2 center coordinates (ICRS J2000)
ra_center = "06h21m14.5s"  # Right Ascension in hms
dec_center = "-47d53m13s"  # Declination in dms

# Convert to SkyCoord object
lops2_center = SkyCoord(ra=ra_center, dec=dec_center, frame='icrs')

# Define the field of view (49° x 49°)
fov_size = 49 * u.deg

# Create a WCS object for plotting in equatorial coordinates
wcs = WCS(naxis=2)
wcs.wcs.crval = [lops2_center.ra.deg, lops2_center.dec.deg]  # Center of the field
wcs.wcs.crpix = [0.5, 0.5]  # Reference pixel
wcs.wcs.cdelt = [-1, 1]  # Pixel scale (deg/pixel)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Projection type

# Plot the LOPS2 field
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection=wcs)
ax.set_title("PLATO LOPS2 Field")
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")

# Add a square representing the FoV
fov_square = SphericalCircle((lops2_center.ra, lops2_center.dec),
                             radius=fov_size / 2,
                             edgecolor='blue', facecolor='none', transform=ax.get_transform('icrs'))
ax.add_patch(fov_square)

# Set gridlines and limits
ax.grid(color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(-fov_size.value / 2, fov_size.value / 2)
ax.set_ylim(-fov_size.value / 2, fov_size.value / 2)

plt.show()
