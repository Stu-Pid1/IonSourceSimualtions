"""
Main ion source simulation class that orchestrates all components.
"""

import numpy as np
from .constants import e, m_e, TANTALUM_WORK_FUNCTION
from .physics.emission import richardson_dushman, electron_velocity_distribution, electron_emission_angle
from .solvers import PoissonSolverCylindrical
from .particles import Particle, ParticleEnsemble, ParticleTracker


class IonSourceSimulation:
    """
    Main simulation class for ion source behavior.

    Combines:
    - Temperature and thermionic emission modeling
    - Poisson equation solving for potential well
    - Particle tracking and dynamics
    """

    def __init__(
        self,
        tube_radius=1.5e-3,
        tube_length=30e-3,
        temperature=2573,
        extraction_voltage=0.0,
        nr=50,
        nz=100,
    ):
        """
        Initialize ion source simulation.

        Parameters
        ----------
        tube_radius : float, optional
            Tube radius [m], default 1.5 mm
        tube_length : float, optional
            Tube length [m], default 30 mm
        temperature : float, optional
            Tube temperature [K], default 2573 K (~2300°C)
        extraction_voltage : float, optional
            Applied extraction voltage [V]
        nr : int, optional
            Number of radial grid points
        nz : int, optional
            Number of axial grid points
        """
        self.tube_radius = tube_radius
        self.tube_length = tube_length
        self.temperature = temperature
        self.extraction_voltage = extraction_voltage

        # Create Poisson solver
        self.solver = PoissonSolverCylindrical(tube_radius, tube_length, nr=nr, nz=nz)

        # Solve initial potential
        self._update_potential()

        # Initialize particle ensemble and tracker
        self.ensemble = ParticleEnsemble()
        self.tracker = ParticleTracker(self.ensemble, field_solver=self._get_field)

        # Store parameters
        self.particle_counter = 0

    def _update_potential(self):
        """Solve Poisson equation for current conditions."""
        # Boundary conditions
        boundary = {
            "r0": 0.0,  # Axis: no boundary condition
            "rR": self.extraction_voltage,  # Tube wall
            "z0": 0.0,  # Entrance
            "zL": 0.0,  # Exit
        }

        self.potential = self.solver.solve(boundary, charge_density=None)

    def _get_field(self, r, z):
        """
        Get electric field at a point.

        Parameters
        ----------
        r : float
            Radial coordinate [m]
        z : float
            Axial coordinate [m]

        Returns
        -------
        E_r, E_z : float
            Electric field components [V/m]
        """
        return self.solver.get_field_at_point(r, z, self.potential)

    def set_temperature(self, temperature):
        """
        Set tube temperature.

        Parameters
        ----------
        temperature : float
            Temperature [K]
        """
        self.temperature = temperature

    def set_extraction_voltage(self, voltage):
        """
        Set extraction voltage.

        Parameters
        ----------
        voltage : float
            Voltage [V]
        """
        self.extraction_voltage = voltage
        self._update_potential()

    def emit_electrons(self, n_electrons=100, emission_area=None, dt=1e-9):
        """
        Emit electrons from the tube wall due to thermionic emission.

        Parameters
        ----------
        n_electrons : int, optional
            Number of electrons to emit
        emission_area : float, optional
            Emitting surface area [m²]. If None, uses tube surface area
        dt : float, optional
            Time step for emission simulation [s]
        """
        if emission_area is None:
            # Surface area of cylindrical tube
            emission_area = 2 * np.pi * self.tube_radius * self.tube_length

        # Calculate emission current
        current = richardson_dushman(self.temperature, TANTALUM_WORK_FUNCTION, emission_area)

        # Generate thermal velocities
        velocities = electron_velocity_distribution(self.temperature, n_samples=n_electrons)
        theta, phi = electron_emission_angle(n_samples=n_electrons)

        # Create electrons
        for i in range(n_electrons):
            # Random position on tube surface
            z_pos = np.random.uniform(0, self.tube_length)
            phi_angle = np.random.uniform(0, 2 * np.pi)

            # Position on tube surface
            x = self.tube_radius * np.cos(phi_angle)
            y = self.tube_radius * np.sin(phi_angle)

            # Velocity components
            v_mag = velocities[i]
            vx = v_mag * np.sin(theta[i]) * np.cos(phi[i])
            vy = v_mag * np.sin(theta[i]) * np.sin(phi[i])
            vz = v_mag * np.cos(theta[i])

            position = np.array([x, y, z_pos])
            velocity = np.array([vx, vy, vz])

            # Create and add electron
            electron = Particle(
                charge=-e, mass=m_e, position=position, velocity=velocity, particle_id=self.particle_counter
            )
            self.particle_counter += 1
            self.ensemble.add_particle(electron)

    def emit_ions(self, n_ions=10, ion_mass_amu=4, ion_charge=1, position_center=None, velocity_spread=100):
        """
        Emit ions from the tube center.

        Parameters
        ----------
        n_ions : int, optional
            Number of ions to emit
        ion_mass_amu : float, optional
            Ion mass [atomic mass units]
        ion_charge : float, optional
            Ion charge [elementary charges]
        position_center : ndarray, optional
            Center position for ion emission [m]
        velocity_spread : float, optional
            Velocity spread [m/s]
        """
        if position_center is None:
            position_center = np.array([0, 0, self.tube_length / 2])

        # Convert mass and charge to SI
        amu_to_kg = 1.66053906660e-27
        ion_mass_kg = ion_mass_amu * amu_to_kg
        ion_charge_C = ion_charge * e

        for i in range(n_ions):
            # Random position near center
            r = np.random.normal(0, self.tube_radius / 5)
            phi = np.random.uniform(0, 2 * np.pi)

            x = position_center[0] + r * np.cos(phi)
            y = position_center[1] + r * np.sin(phi)
            z = position_center[2] + np.random.normal(0, self.tube_length / 10)

            # Random velocity
            vx = np.random.normal(0, velocity_spread)
            vy = np.random.normal(0, velocity_spread)
            vz = np.random.normal(velocity_spread / 2, velocity_spread / 4)

            position = np.array([x, y, z])
            velocity = np.array([vx, vy, vz])

            ion = Particle(
                charge=ion_charge_C, mass=ion_mass_kg, position=position, velocity=velocity, particle_id=self.particle_counter
            )
            self.particle_counter += 1
            self.ensemble.add_particle(ion)

    def run(self, duration, dt, n_electron_emissions=10, use_coulomb=False):
        """
        Run the simulation.

        Parameters
        ----------
        duration : float
            Total simulation time [s]
        dt : float
            Integration time step [s]
        n_electron_emissions : int, optional
            Number of times to emit electrons during simulation
        use_coulomb : bool, optional
            Include particle-particle Coulomb interactions
        """
        emission_interval = duration / n_electron_emissions

        # Emit initial electrons
        self.emit_electrons(n_electrons=100)

        # Run with periodic emissions
        n_steps = int(duration / dt)
        next_emission = emission_interval

        for step in range(n_steps):
            current_time = step * dt

            # Emit electrons periodically
            if current_time >= next_emission:
                self.emit_electrons(n_electrons=50)
                next_emission += emission_interval

            # Step simulation
            self.tracker.step(dt, use_coulomb=use_coulomb)

    def get_summary(self):
        """Get simulation summary."""
        summary = {
            "tube_radius": self.tube_radius * 1000,  # mm
            "tube_length": self.tube_length * 1000,  # mm
            "temperature": self.temperature,  # K
            "extraction_voltage": self.extraction_voltage,  # V
            "n_particles": len(self.ensemble.particles),
            "simulation_time": self.tracker.time,
            "n_steps": self.tracker.step_count,
        }
        return summary

    def __repr__(self):
        """String representation."""
        return (
            f"IonSourceSimulation("
            f"radius={self.tube_radius*1000:.1f}mm, "
            f"length={self.tube_length*1000:.1f}mm, "
            f"T={self.temperature:.0f}K, "
            f"n_particles={len(self.ensemble.particles)})"
        )
