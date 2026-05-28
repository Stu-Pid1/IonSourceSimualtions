# Ion Source Simulations

A physics simulation package for modeling ion source behavior with hot surface ion sources used in radioactive ion beam (RIB) generation.

## Overview

This project simulates the complex dynamics of electrons and ions within a heated tantalum ion source tube. The simulation accounts for:

- **Thermal electron emission** from a heated surface (Richardson-Dushman model)
- **Electric potential** distribution in cylindrical geometry (Poisson equation)
- **Particle dynamics** under electromagnetic forces
- **Variable temperature** control (uniform across tube, expandable to spatial variation)
- **Large-scale particle tracking** for high electron/ion populations

### Physical System

The baseline system models:
- **Tube material**: Tantalum
- **Temperature**: 2300°C (2573 K) - tunable
- **Tube radius**: ~3 mm (variable)
- **Tube length**: 20-50 mm (variable)
- **Thermally emitted electrons**: From heated surface
- **Ion generation**: Variable ion species and energy

## Installation

### Requirements
- Python 3.9+
- NumPy, SciPy, Matplotlib
- Numba for performance (optional)
- CuPy for GPU acceleration (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/Stu-Pid1/IonSourceSimualtions.git
cd IonSourceSimualtions

# Install in development mode
pip install -e .

# Or with optional dependencies
pip install -e ".[gpu,distributed,dev]"
```

## Quick Start

```python
from ionsource import IonSourceSimulation
from ionsource.visualization import plot_potential, plot_electric_field

# Create simulation
sim = IonSourceSimulation(
    tube_radius=1.5e-3,      # 3 mm diameter
    tube_length=30e-3,       # 30 mm length
    temperature=2573,        # 2300°C in Kelvin
    extraction_voltage=1000  # 1 kV extraction field
)

# Emit particles
sim.emit_electrons(n_electrons=500)
sim.emit_ions(n_ions=100, ion_mass_amu=4)

# Run simulation
sim.run(duration=1e-7, dt=1e-10)  # 100 ns with 0.1 ns steps

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2)
plot_potential(sim.solver, sim.potential, ax=axes[0])
plot_electric_field(sim.solver, sim.potential, ax=axes[1])
plt.show()
```

See `examples/basic_simulation.py` for more detailed examples.

## Project Structure

```
ionsource/
├── __init__.py                 # Package initialization
├── constants.py               # Physical constants
├── simulation.py              # Main simulation class
│
├── physics/                   # Physics models
│   ├── emission.py           # Thermionic emission (Richardson-Dushman)
│   └── forces.py             # Electromagnetic forces
│
├── solvers/                  # Equation solvers
│   └── __init__.py           # Poisson solver (cylindrical coordinates)
│
├── particles/                # Particle tracking
│   └── __init__.py           # Particle class and ensemble
│
└── visualization/            # Plotting utilities
    └── __init__.py          # Field and trajectory visualization

examples/
└── basic_simulation.py       # Example usage script
```

## Features

### Current Implementation

✅ **Physics**
- Richardson-Dushman thermionic emission
- Maxwell-Boltzmann velocity distribution
- Lorentz force (electric field only)
- Coulomb interactions (optional)

✅ **Numerics**
- Finite difference Poisson solver (cylindrical coordinates)
- Velocity Verlet integration
- Particle tracking and ensemble management

✅ **Visualization**
- Potential and field contour plots
- Electric field vector fields
- Particle trajectory plots
- Phase space distributions

### Planned Enhancements

🔄 **Temperature Variation**
- Spatial temperature gradients along tube axis
- Time-dependent temperature changes

🔄 **Performance Optimization**
- Numba JIT compilation for hot loops
- GPU acceleration (CuPy backend)
- Distributed computing (Dask/MPI)

🔄 **Advanced Physics**
- Ion-electron collisions and ionization
- Charge exchange reactions
- Magnetic field effects
- Space charge limited current

🔄 **Output & Analysis**
- HDF5 trajectory storage
- Statistical analysis tools
- Transport coefficient calculation

## Usage Examples

### Basic Simulation
```python
sim = IonSourceSimulation()
sim.emit_electrons(500)
sim.run(duration=1e-7, dt=1e-10)
```

### With Extraction Voltage
```python
sim = IonSourceSimulation(extraction_voltage=2000)  # 2 kV
sim.emit_electrons(1000)
sim.run(duration=1e-7, dt=1e-10)
```

### Variable Temperature
```python
sim = IonSourceSimulation(temperature=2273)  # 2000°C
sim.run(duration=1e-7, dt=1e-10)

# Later: change temperature
sim.set_temperature(2800)  # 2527°C
```

### With Coulomb Interactions
```python
sim = IonSourceSimulation()
sim.emit_electrons(500)
sim.run(duration=1e-8, dt=1e-11, use_coulomb=True)
```

## Physical Constants & Models

### Richardson-Dushman Equation
```
J = A * T² * exp(-φ / (k_B * T))
```
- J: emission current density
- A: Richardson constant (~120 A/(m²·K²))
- T: temperature
- φ: work function (4.25 eV for tantalum)

### Electric Field (Cylindrical)
Solves Poisson equation:
```
∇²φ = -ρ / ε₀
```
With finite differences: `1/r * d(r * dφ/dr)/dr + d²φ/dz² = -ρ/ε₀`

### Particle Motion
Velocity Verlet integration:
```
x(t+Δt) = x(t) + v(t)*Δt + 0.5*a(t)*Δt²
v(t+Δt) = v(t) + 0.5*(a(t) + a(t+Δt))*Δt
```

## Performance Notes

For typical simulations:
- **CPU**: ~1-10k particles, microsecond timescales
- **GPU**: Planned, 100k+ particles
- **Distributed**: Planned for parameter sweeps

Computation scales as O(N²) with Coulomb interactions enabled, O(N) without.

## Contributing

Contributions welcome! Areas of interest:
- Performance optimization
- Additional physics models
- Improved visualization
- Testing and validation

## License

MIT License - see LICENSE file

## References

### Ion Source Physics
- Geller, R., et al. (1986). "Hot-cavity positive-ion source" - Nuclear Instruments and Methods
- Wenander, F. (2013). "Developments in ion sources for exotic beams" - Nuclear Instruments and Methods

### Numerical Methods
- Press, W.H., et al. (2007). "Numerical Recipes" - Cambridge University Press
- LeVeque, R.J. (2007). "Finite Difference Methods for Ordinary and Partial Differential Equations"

## Contact

Questions or suggestions? Open an issue on GitHub.
