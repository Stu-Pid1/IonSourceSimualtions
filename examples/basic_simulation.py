#!/usr/bin/env python
"""
Example ion source simulation script.

This demonstrates basic usage of the ion source simulation package:
1. Creating a simulation
2. Running particle tracking
3. Visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from ionsource import IonSourceSimulation
from ionsource.visualization import (
    plot_potential,
    plot_electric_field,
    plot_particle_trajectories,
    plot_particle_distribution,
)


def example_basic_simulation():
    """Run a basic ion source simulation."""
    print("=" * 60)
    print("Ion Source Simulation - Basic Example")
    print("=" * 60)

    # Create simulation with default parameters
    # Tube: 3 mm radius, 30 mm length
    # Temperature: 2573 K (2300°C)
    # Extraction voltage: 0 V
    sim = IonSourceSimulation(
        tube_radius=1.5e-3,  # 3 mm diameter
        tube_length=30e-3,  # 30 mm length
        temperature=2573,  # ~2300°C in Kelvin
        extraction_voltage=0.0,  # No extraction yet
    )

    print(f"\nSimulation initialized:")
    print(sim)

    # Emit electrons and ions
    print("\nEmitting particles...")
    sim.emit_electrons(n_electrons=200)
    sim.emit_ions(n_ions=50, ion_mass_amu=4, ion_charge=2)

    print(f"Total particles: {len(sim.ensemble.particles)}")

    # Run simulation
    print("\nRunning simulation...")
    duration = 1e-7  # 100 ns
    dt = 1e-10  # 0.1 ns time step
    sim.run(duration, dt, n_electron_emissions=5, use_coulomb=False)

    # Print summary
    print("\nSimulation Summary:")
    summary = sim.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")

    return sim


def example_with_extraction_voltage():
    """Run simulation with extraction voltage."""
    print("\n" + "=" * 60)
    print("Ion Source Simulation - With Extraction Voltage")
    print("=" * 60)

    # Create simulation with extraction voltage
    sim = IonSourceSimulation(
        tube_radius=1.5e-3,
        tube_length=30e-3,
        temperature=2573,
        extraction_voltage=1000.0,  # 1 kV extraction
    )

    print(f"\nSimulation with {sim.extraction_voltage:.0f} V extraction voltage")

    # Emit and run
    sim.emit_electrons(n_electrons=150)
    sim.run(duration=1e-7, dt=1e-10, n_electron_emissions=3, use_coulomb=False)

    return sim


def visualize_results(sim):
    """Create visualization plots."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Electric potential
    plot_potential(sim.solver, sim.potential, ax=axes[0, 0])

    # Plot 2: Electric field
    plot_electric_field(sim.solver, sim.potential, ax=axes[0, 1])

    # Plot 3: Particle trajectories (r-z view)
    plot_particle_trajectories(sim.tracker, ax=axes[1, 0], dim="rz")

    # Plot 4: Current particle distribution
    positions = sim.ensemble.get_positions()
    plot_particle_distribution(positions, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("ion_source_simulation.png", dpi=150, bbox_inches="tight")
    print("Saved: ion_source_simulation.png")

    plt.show()


if __name__ == "__main__":
    # Run basic simulation
    sim1 = example_basic_simulation()

    # Run with extraction voltage
    sim2 = example_with_extraction_voltage()

    # Visualize results from first simulation
    visualize_results(sim1)

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
