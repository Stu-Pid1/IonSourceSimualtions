"""
Thermionic emission models for hot surface ion sources.
"""

import numpy as np
from ..constants import k_B, e, TANTALUM_WORK_FUNCTION


def richardson_dushman(temperature, work_function, area):
    """
    Calculate thermionic emission current using the Richardson-Dushman equation.

    The Richardson-Dushman equation describes the emission of electrons from
    a heated metal surface:

        J = A * T^2 * exp(-phi / (k_B * T))

    where:
    - J is the emission current density [A/m^2]
    - A is Richardson's constant [A/(m^2*K^2)]
    - T is absolute temperature [K]
    - phi is the work function [eV]
    - k_B is Boltzmann constant [J/K]

    Parameters
    ----------
    temperature : float or array
        Absolute temperature [K]
    work_function : float
        Work function [eV]
    area : float
        Emitting surface area [m^2]

    Returns
    -------
    current : float or array
        Total emission current [A]
    """
    # Richardson constant for tantalum (commonly used value)
    A_richardson = 120e4  # [A/(m^2*K^2)] for tungsten/tantalum

    # Convert work function from eV to Joules
    phi_joules = work_function * e

    # Calculate current density
    current_density = A_richardson * temperature**2 * np.exp(
        -phi_joules / (k_B * temperature)
    )

    # Total current
    current = current_density * area

    return current


def thermionic_current(temperature, work_function, area):
    """
    Calculate total thermionic emission current.

    This is a wrapper around richardson_dushman for clarity.

    Parameters
    ----------
    temperature : float or array
        Absolute temperature [K]
    work_function : float
        Work function [eV]
    area : float
        Emitting surface area [m^2]

    Returns
    -------
    current : float or array
        Total emission current [A]
    """
    return richardson_dushman(temperature, work_function, area)


def electron_velocity_distribution(temperature, n_samples=1000):
    """
    Generate electron velocity distribution based on temperature.

    Uses Maxwell-Boltzmann distribution for thermal velocities.

    Parameters
    ----------
    temperature : float
        Absolute temperature [K]
    n_samples : int, optional
        Number of velocity samples to generate

    Returns
    -------
    velocities : ndarray
        Electron velocities [m/s] with shape (n_samples,)
    """
    from ..constants import m_e

    # Thermal velocity scale
    v_thermal = np.sqrt(k_B * temperature / m_e)

    # Sample from Maxwell-Boltzmann distribution
    # Generate three components of velocity
    vx = np.random.normal(0, v_thermal / np.sqrt(3), n_samples)
    vy = np.random.normal(0, v_thermal / np.sqrt(3), n_samples)
    vz = np.random.normal(0, v_thermal / np.sqrt(3), n_samples)

    # Calculate magnitude
    velocities = np.sqrt(vx**2 + vy**2 + vz**2)

    return velocities


def electron_emission_angle(n_samples=1000):
    """
    Generate random emission angles for thermionic electrons.

    Electrons are emitted isotropically in a hemisphere above the surface.

    Parameters
    ----------
    n_samples : int, optional
        Number of angles to generate

    Returns
    -------
    theta : ndarray
        Polar angles [rad]
    phi : ndarray
        Azimuthal angles [rad]
    """
    # Isotropic emission in upper hemisphere (0 to pi/2)
    cos_theta = np.random.uniform(0, 1, n_samples)
    theta = np.arccos(cos_theta)

    # Full azimuthal range
    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    return theta, phi
