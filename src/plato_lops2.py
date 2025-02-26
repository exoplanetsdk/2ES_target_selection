import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

# Define the LOPS2 field parameters
field_center = {"l": 255.9375, "b": -24.62432}  # Galactic coordinates of the center
field_radius = 28.1  # Radius of the field in degrees

# Define camera coverage areas (in degrees^2)
camera_areas = {
    "24 cameras": 325,
    "18 cameras": 153,
    "12 cameras": 847,
    "6 cameras": 824
}

# Healpix resolution
nside = 128  # Adjust for resolution; higher values give finer detail

# Create a Healpix map for the field
npix = hp.nside2npix(nside)
field_map = np.zeros(npix)

# Convert galactic coordinates of the center to Healpix indices
theta_center = np.radians(90 - field_center["b"])  # Convert latitude to colatitude
phi_center = np.radians(field_center["l"])  # Convert longitude to radians
vec_center = hp.ang2vec(theta_center, phi_center)

# Assign different coverage regions based on camera coverage
for i, (camera, area) in enumerate(camera_areas.items()):
    radius = np.sqrt(area / np.pi)  # Approximate radius for each camera coverage
    pixels = hp.query_disc(nside, vec_center, np.radians(radius))
    field_map[pixels] = i + 1

# Plot the field
hp.mollview(
    field_map,
    coord="G",
    title="LOPS2 Field with Camera Coverage",
    cmap="YlGnBu",
    unit="Camera Coverage",
    notext=True
)

# Add a legend for the camera coverage
legend_labels = list(camera_areas.keys())
legend_colors = [plt.cm.YlGnBu(i / len(camera_areas)) for i in range(len(camera_areas))]
for label, color in zip(legend_labels, legend_colors):
    plt.scatter([], [], c=[color], label=label)

plt.legend(loc="lower left", fontsize="small", title="Coverage")
plt.show()
