"""
Visualization utilities for ion source simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


def plot_potential(solver, potential, ax=None, cmap="viridis"):
    """
    Plot electric potential distribution.

    Parameters
    ----------
    solver : PoissonSolverCylindrical
        Poisson solver with grid information
    potential : ndarray
        Potential values, shape (nr, nz)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str, optional
        Colormap name

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image object
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Create mesh in mm for better readability
    R_mm = solver.R * 1000
    Z_mm = solver.Z * 1000

    im = ax.contourf(Z_mm, R_mm, potential, levels=20, cmap=cmap)
    ax.set_xlabel("Axial Position [mm]")
    ax.set_ylabel("Radial Position [mm]")
    ax.set_title("Electric Potential [V]")
    plt.colorbar(im, ax=ax, label="Potential [V]")

    return im


def plot_electric_field(solver, potential, ax=None, skip=2, cmap="plasma"):
    """
    Plot electric field vectors and magnitude.

    Parameters
    ----------
    solver : PoissonSolverCylindrical
        Poisson solver
    potential : ndarray
        Potential values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    skip : int, optional
        Skip factor for vector plotting (plot every skip-th vector)
    cmap : str, optional
        Colormap for magnitude

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    E_r, E_z = solver.get_electric_field(potential)
    E_mag = np.sqrt(E_r**2 + E_z**2)

    # Plot field magnitude as contours
    R_mm = solver.R * 1000
    Z_mm = solver.Z * 1000

    levels = np.linspace(np.min(E_mag), np.max(E_mag), 15)
    im = ax.contourf(Z_mm, R_mm, E_mag, levels=levels, cmap=cmap)
    plt.colorbar(im, ax=ax, label="Field Magnitude [V/m]")

    # Plot field vectors
    Q = ax.quiver(
        Z_mm[::skip, ::skip],
        R_mm[::skip, ::skip],
        E_z[::skip, ::skip],
        E_r[::skip, ::skip],
        scale=np.max(E_mag) * 5,
    )

    ax.set_xlabel("Axial Position [mm]")
    ax.set_ylabel("Radial Position [mm]")
    ax.set_title("Electric Field Distribution")

    return ax


def plot_particle_trajectories(tracker, particle_indices=None, ax=None, dim="rz"):
    """
    Plot particle trajectories.

    Parameters
    ----------
    tracker : ParticleTracker
        Particle tracker with simulation history
    particle_indices : list, optional
        Indices of particles to plot (all if None)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    dim : str, optional
        Projection: 'rz' for radial-axial, '3d' for 3D

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if particle_indices is None:
        particle_indices = range(len(tracker.ensemble.particles))

    if ax is None:
        if dim == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()

    for idx in particle_indices:
        if idx >= len(tracker.ensemble.particles):
            continue

        particle = tracker.ensemble.particles[idx]
        history = particle.history["position"]

        if len(history) == 0:
            continue

        positions = np.array(history)

        if dim == "rz":
            # Radial-axial projection
            r = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2) * 1000
            z = positions[:, 2] * 1000
            ax.plot(z, r, label=f"Particle {particle.particle_id}", alpha=0.7)

        elif dim == "3d":
            ax.plot(
                positions[:, 0] * 1000,
                positions[:, 1] * 1000,
                positions[:, 2] * 1000,
                label=f"Particle {particle.particle_id}",
                alpha=0.7,
            )

    if dim == "rz":
        ax.set_xlabel("Axial Position [mm]")
        ax.set_ylabel("Radial Position [mm]")
        ax.set_title("Particle Trajectories (r-z projection)")
    else:
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.set_title("Particle Trajectories (3D)")

    ax.legend()

    return ax


def plot_particle_distribution(positions, ax=None):
    """
    Plot current particle positions (phase space).

    Parameters
    ----------
    positions : ndarray
        Current particle positions, shape (n_particles, 3)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    if len(positions) > 0:
        r = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2) * 1000
        z = positions[:, 2] * 1000

        ax.scatter(z, r, alpha=0.5, s=10)
        ax.set_xlabel("Axial Position [mm]")
        ax.set_ylabel("Radial Position [mm]")
        ax.set_title("Current Particle Distribution")
    else:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)

    return ax


__all__ = [
    "plot_potential",
    "plot_electric_field",
    "plot_particle_trajectories",
    "plot_particle_distribution",
]
