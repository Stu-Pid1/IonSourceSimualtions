"""
Force calculations for particles in electromagnetic fields.
"""

import numpy as np
from ..constants import e, m_e, k_e


def electric_force(charge, electric_field):
    """
    Calculate electric force on a particle.

    F = q * E

    Parameters
    ----------
    charge : float or array
        Particle charge [C]
    electric_field : ndarray
        Electric field vector [V/m], shape (..., 3) for 3D components

    Returns
    -------
    force : ndarray
        Force vector [N], same shape as electric_field
    """
    return charge * electric_field


def coulomb_force(q1, q2, r_vec):
    """
    Calculate Coulomb force between two point charges.

    F = k_e * q1 * q2 / r^2 * r_hat

    Parameters
    ----------
    q1, q2 : float
        Charges [C]
    r_vec : ndarray
        Separation vector [m], shape (..., 3)

    Returns
    -------
    force : ndarray
        Force vector [N] on charge q1, same shape as r_vec
    """
    r_mag = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat = r_vec / r_mag

    # Avoid division by zero
    r_mag = np.maximum(r_mag, 1e-50)

    force_mag = k_e * q1 * q2 / r_mag**2
    force = force_mag * r_hat

    return force


def lorentz_force(charge, velocity, electric_field, magnetic_field=None):
    """
    Calculate Lorentz force on a charged particle.

    F = q * (E + v × B)

    Parameters
    ----------
    charge : float or array
        Particle charge [C]
    velocity : ndarray
        Velocity vector [m/s], shape (..., 3)
    electric_field : ndarray
        Electric field [V/m], shape (..., 3)
    magnetic_field : ndarray, optional
        Magnetic field [T], shape (..., 3). If None, only electric force is calculated.

    Returns
    -------
    force : ndarray
        Force vector [N], shape (..., 3)
    """
    # Electric force component
    f_electric = charge * electric_field

    # Magnetic force component if provided
    if magnetic_field is not None:
        f_magnetic = charge * np.cross(velocity, magnetic_field, axisa=-1, axisb=-1)
        f_total = f_electric + f_magnetic
    else:
        f_total = f_electric

    return f_total


def particle_acceleration(force, mass):
    """
    Calculate acceleration from force.

    a = F / m

    Parameters
    ----------
    force : ndarray
        Force vector [N], shape (..., 3)
    mass : float
        Particle mass [kg]

    Returns
    -------
    acceleration : ndarray
        Acceleration vector [m/s^2], same shape as force
    """
    return force / mass


def electron_acceleration(force):
    """
    Calculate electron acceleration.

    Convenience function using electron mass.

    Parameters
    ----------
    force : ndarray
        Force vector [N], shape (..., 3)

    Returns
    -------
    acceleration : ndarray
        Acceleration vector [m/s^2], same shape as force
    """
    return particle_acceleration(force, m_e)
