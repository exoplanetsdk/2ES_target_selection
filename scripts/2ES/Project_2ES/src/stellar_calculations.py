import numpy as np
from scipy import optimize

#------------------------------------------------------------------------------------------------

def calculate_habitable_zone(T_eff, L_ratio):
    """
    Calculate the habitable zone boundary based on Kopparapu et al. (2013) Equations 2 & 3,
    using the "Moist Greenhouse" and "Maximum Greenhouse" limits.
    
    Parameters:
    T_eff (float): Effective temperature of the star in Kelvin
    
    Returns:
    float: Distance of the habitable zone boundary in AU
    """
    
    def calculate_s_eff(S_eff_sun, a, b, c, d, T_star):
        return S_eff_sun + a*T_star + b*T_star**2 + c*T_star**3 + d*T_star**4

    T_star = T_eff - 5780

    '''
    ------------------------------------------------------------------------------------------------
    Inner Habitable Zone
    ------------------------------------------------------------------------------------------------
    "Recent Venus": 
    S_eff_sun, a, b, c, d = [1.7753, 1.4316E-4, 2.9875E-9, -7.5702E-12, -1.1635E-15]
    "Runaway Greenhouse": 
    S_eff_sun, a, b, c, d = [1.0512, 1.3242E-4, 1.5418E-8, -7.9895E-12, -1.8328E-15]
    "Moist Greenhouse":
    S_eff_sun, a, b, c, d = [1.0140, 8.1774E-5, 1.7063E-9, -4.3241E-12, -6.6462E-16]

    ------------------------------------------------------------------------------------------------
    Outer Habitable Zone
    ------------------------------------------------------------------------------------------------
    "Maximum Greenhouse":
    S_eff_sun, a, b, c, d = [0.3438, 5.8942E-5, 1.6558E-9, -3.0045E-12, -5.2983E-16]
    "Early Mars":
    S_eff_sun, a, b, c, d = [0.3179, 5.4513E-5, 1.5313E-9, -2.7786E-12, -4.8997E-16]
    '''

    # Coefficients for "Moist Greenhouse" and "Maximum Greenhouse"
    coefficients = {
        "Moist Greenhouse": [1.0140, 8.1774E-5, 1.7063E-9, -4.3241E-12, -6.6462E-16],
        "Maximum Greenhouse": [0.3438, 5.8942E-5, 1.6558E-9, -3.0045E-12, -5.2983E-16]
    }

    # Calculate inner and outer distances
    distance_inner = np.sqrt(L_ratio / calculate_s_eff(*coefficients["Moist Greenhouse"], T_star))
    distance_outer = np.sqrt(L_ratio / calculate_s_eff(*coefficients["Maximum Greenhouse"], T_star))

    return 0.5 * (distance_inner + distance_outer)

#------------------------------------------------------------------------------------------------

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_earth = 5.97e24  # Earth mass in kg
M_sun = 1.989e30  # Solar mass in kg
AU = 1.496e11  # 1 AU in meters

def calculate_hz_detection_limit(K, stellar_mass, orbital_radius):
    """
    Calculate the minimum detectable planet mass in the Habitable Zone.
    
    :param K: RV precision in m/s
    :param stellar_mass: Mass of the star in solar masses
    :param orbital_radius: Orbital radius (HZ limit) in AU
    :return: Minimum detectable planet mass in Earth masses or np.nan if calculation fails
    """
    try:
        stellar_mass_kg = stellar_mass * M_sun
        orbital_radius_m = orbital_radius * AU
        
        if K <= 0 or stellar_mass_kg <= 0 or orbital_radius_m <= 0:
            return np.nan
        
        def equation(m_p):
            return K - (G**(1/2) * orbital_radius_m**(-1/2) * m_p * (stellar_mass_kg + m_p)**(-1/2))
        
        # Use numerical method to solve the equation
        planet_mass_kg = optimize.brentq(equation, 0, stellar_mass_kg)
        
        return planet_mass_kg / M_earth
    except:
        return np.nan
        
#------------------------------------------------------------------------------------------------

def calculate_hz_detection_limit_simplify(K, stellar_mass, orbital_radius):
    """
    Calculate the planet mass given RV amplitude, stellar mass, and orbital radius.

    Assumptions: M_star >> M_planet
    """
    # Gravitational constant in SI units (m^3 kg^-1 s^-2)
    stellar_mass_kg = stellar_mass * M_sun
    orbital_radius_m = orbital_radius * AU    

    # Calculate planet mass using the simplified formula
    planet_mass = K * (orbital_radius_m * stellar_mass_kg / G) ** 0.5 / M_earth

    return planet_mass    


#--------------------------------------------------------------------------------------------------

