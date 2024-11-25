#************************************************************************************
# Compute HZ runaway greenhouse limit and mass of planet at location
#************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from astropy import constants as cc
from astropy import units as uu

#************************************************************************************
# Output files.


#************************************************************************************
# Coeffcients to be used in the analytical expression to calculate habitable zone flux 
# boundaries

# initialize flux and distance
seff = [0,0,0,0,0,0]
dist = [0,0,0,0,0,0]
# coefficients
seffsun  = [1.776,1.107, 0.356, 0.320, 1.188, 0.99] 
a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]

#************************************************************************************
# Read exoplanet archive data
dat = np.genfromtxt('/home/hobson/Documents/Second_Earth_initiative/hzaux.csv',
      delimiter = ',', names=True)

Teff = dat['st_teff']
Lum = dat['st_lum']
Mass = dat['st_mass']
rvnoise = dat['rv_noise']

#************************************************************************************
# Calculating HZ fluxes for stars with 2600 K < T_eff < 7200 K. 

# general HZ storage
dist_tenthmeRunaway = []
mass_planet = []
mass_planet_val = []

# loop over the systems
for j in range(len(Teff)):
  if 2600<Teff[j] and Teff[j]<7200:
    teff = Teff[j]
    tstar = teff - 5780.0
    lum = 10**Lum[j]
    for i in range(len(a)):
      # compute fluxes
      seff[i] = seffsun[i] + a[i]*tstar + b[i]*tstar**2 + c[i]*tstar**3 + d[i]*tstar**4
      # compute distances
      dist[i] = lum**0.5/seff[i]
    # save distances
    dist_tenthmeRunaway.append(dist[5])
    # compute mass
    K = rvnoise[j]*uu.m/uu.s
    mstar = Mass[j]*cc.M_sun
    ax = dist[5]*uu.au
    mpl = K*np.sqrt(ax.to(uu.m)*mstar/cc.G)
    mass_planet.append(mpl.to(uu.M_earth))
    mass_planet_val.append(mpl.to(uu.M_earth).value)

towrite = np.column_stack((dat['Gaia_name'], mass_planet_val))

np.savetxt('hz_masses.txt', towrite)