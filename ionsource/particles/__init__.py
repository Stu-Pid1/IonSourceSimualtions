"""
Particle tracking and dynamics simulation.
"""

import numpy as np
from tqdm import tqdm


class Particle:
    """Represents a single particle."""

    def __init__(self, charge, mass, position, velocity, particle_id=0):
        """
        Initialize a particle.

        Parameters
        ----------
        charge : float
            Particle charge [C]
        mass : float
            Particle mass [kg]
        position : ndarray
            Initial position [m], shape (3,)
        velocity : ndarray
            Initial velocity [m/s], shape (3,)
        particle_id : int, optional
            Unique particle identifier
        """
        self.charge = charge
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.particle_id = particle_id
        self.history = {
            "position": [self.position.copy()],
            "velocity": [self.velocity.copy()],
        }

    def update_acceleration(self, force):
        """Update acceleration from force."""
        self.acceleration = force / self.mass

    def step_velocity_verlet(self, dt):
        """
        Update velocity and position using velocity Verlet algorithm.

        v(t+dt) = v(t) + a(t) * dt + 1/2 * a(t+dt) * dt
        x(t+dt) = x(t) + v(t) * dt + 1/2 * a(t) * dt²

        Parameters
        ----------
        dt : float
            Time step [s]
        """
        # Half step for velocity
        v_half = self.velocity + 0.5 * self.acceleration * dt

        # Full step for position
        self.position += v_half * dt

        # Acceleration will be updated by solver
        # v_half will be combined with new acceleration in next step

    def finalize_velocity_verlet(self, new_acceleration, dt):
        """
        Finalize velocity Verlet update with new acceleration.

        Parameters
        ----------
        new_acceleration : ndarray
            New acceleration [m/s²], shape (3,)
        dt : float
            Time step [s]
        """
        self.velocity += 0.5 * (self.acceleration + new_acceleration) * dt
        self.acceleration = new_acceleration


class ParticleEnsemble:
    """Manages a collection of particles."""

    def __init__(self, particles=None):
        """
        Initialize particle ensemble.

        Parameters
        ----------
        particles : list of Particle, optional
            Initial list of particles
        """
        self.particles = particles if particles is not None else []

    def add_particle(self, particle):
        """Add a particle to the ensemble."""
        self.particles.append(particle)

    def add_particles(self, particles):
        """Add multiple particles."""
        self.particles.extend(particles)

    def __len__(self):
        """Number of particles."""
        return len(self.particles)

    def get_positions(self):
        """Get all particle positions."""
        if len(self.particles) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([p.position for p in self.particles])

    def get_velocities(self):
        """Get all particle velocities."""
        if len(self.particles) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([p.velocity for p in self.particles])

    def get_charges(self):
        """Get all particle charges."""
        return np.array([p.charge for p in self.particles])

    def get_masses(self):
        """Get all particle masses."""
        return np.array([p.mass for p in self.particles])

    def remove_particle(self, idx):
        """Remove particle by index."""
        self.particles.pop(idx)

    def remove_particles(self, indices):
        """Remove multiple particles by indices (in reverse order to maintain indices)."""
        for idx in sorted(indices, reverse=True):
            self.remove_particle(idx)


class ParticleTracker:
    """Integrates particle equations of motion."""

    def __init__(self, ensemble, field_solver=None):
        """
        Initialize particle tracker.

        Parameters
        ----------
        ensemble : ParticleEnsemble
            Particle ensemble to track
        field_solver : callable, optional
            Function to compute electric field at position (r, z)
        """
        self.ensemble = ensemble
        self.field_solver = field_solver
        self.time = 0.0
        self.step_count = 0

    def compute_forces(self, use_coulomb=False):
        """
        Compute forces on all particles.

        Parameters
        ----------
        use_coulomb : bool, optional
            Include particle-particle Coulomb interactions
        """
        from .physics.forces import electric_force, coulomb_force, electron_acceleration

        positions = self.ensemble.get_positions()
        charges = self.ensemble.get_charges()

        # External electric field forces
        if self.field_solver is not None:
            for i, particle in enumerate(self.ensemble.particles):
                r = np.sqrt(particle.position[0] ** 2 + particle.position[1] ** 2)
                z = particle.position[2]

                E_r, E_z = self.field_solver(r, z)

                # Convert to Cartesian
                if r > 1e-10:
                    cos_theta = particle.position[0] / r
                    sin_theta = particle.position[1] / r
                    E_x = E_r * cos_theta
                    E_y = E_r * sin_theta
                else:
                    E_x, E_y = 0, 0

                E_field = np.array([E_x, E_y, E_z])
                force = electric_force(particle.charge, E_field)
                particle.update_acceleration(force)

        # Particle-particle Coulomb interactions
        if use_coulomb and len(self.ensemble.particles) > 1:
            for i, particle_i in enumerate(self.ensemble.particles):
                force_total = np.zeros(3)
                for j, particle_j in enumerate(self.ensemble.particles):
                    if i != j:
                        r_vec = particle_j.position - particle_i.position
                        force = coulomb_force(
                            particle_i.charge, particle_j.charge, r_vec.reshape(1, 3)
                        )
                        force_total += force[0]
                particle_i.acceleration = force_total / particle_i.mass

    def step(self, dt, use_coulomb=False):
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step [s]
        use_coulomb : bool, optional
            Include Coulomb interactions
        """
        # Compute initial forces
        self.compute_forces(use_coulomb=use_coulomb)

        # Update velocities and positions (first half of Verlet)
        for particle in self.ensemble.particles:
            particle.step_velocity_verlet(dt)

        # Compute new forces at new positions
        self.compute_forces(use_coulomb=use_coulomb)

        # Finalize velocity updates
        for particle in self.ensemble.particles:
            particle.finalize_velocity_verlet(particle.acceleration, dt)

        # Record history
        for particle in self.ensemble.particles:
            particle.history["position"].append(particle.position.copy())
            particle.history["velocity"].append(particle.velocity.copy())

        self.time += dt
        self.step_count += 1

    def run(self, duration, dt, use_coulomb=False, remove_boundary=True, boundary_radius=None):
        """
        Run simulation for specified duration.

        Parameters
        ----------
        duration : float
            Total simulation duration [s]
        dt : float
            Time step [s]
        use_coulomb : bool, optional
            Include Coulomb interactions
        remove_boundary : bool, optional
            Remove particles that leave the domain
        boundary_radius : float, optional
            Radial boundary for particle removal [m]
        """
        n_steps = int(duration / dt)

        for _ in tqdm(range(n_steps), desc="Simulating"):
            self.step(dt, use_coulomb=use_coulomb)

            if remove_boundary:
                indices_remove = []
                for i, particle in enumerate(self.ensemble.particles):
                    r = np.sqrt(particle.position[0] ** 2 + particle.position[1] ** 2)
                    if particle.position[2] < 0 or particle.position[2] > 0.05:  # Typical tube
                        indices_remove.append(i)
                    elif boundary_radius is not None and r > boundary_radius:
                        indices_remove.append(i)
                self.ensemble.remove_particles(indices_remove)
