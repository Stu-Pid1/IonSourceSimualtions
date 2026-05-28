"""
Physical constants used in ion source simulations.
"""

import numpy as np

# Physical constants (SI units)
k_B = 1.380649e-23  # Boltzmann constant [J/K]
e = 1.602176634e-19  # Elementary charge [C]
m_e = 9.1093837015e-31  # Electron mass [kg]
m_p = 1.67262192369e-27  # Proton mass [kg]
epsilon_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
c = 299792458  # Speed of light [m/s]

# Derived constants
k_e = 1.0 / (4 * np.pi * epsilon_0)  # Coulomb constant [N*m^2/C^2]

# Material properties
TANTALUM_WORK_FUNCTION = 4.25  # eV
TANTALUM_DENSITY = 16650  # kg/m^3

# Ion beam related
# Common ions used in radioactive ion beam generation
ION_MASSES = {
    'p': 1.007825 * 931.494e6,  # Proton [eV/c^2]
    'n': 1.008665 * 931.494e6,  # Neutron [eV/c^2]
    '3He': 3.016029 * 931.494e6,  # Helium-3 [eV/c^2]
    '4He': 4.002603 * 931.494e6,  # Helium-4 [eV/c^2]
}
