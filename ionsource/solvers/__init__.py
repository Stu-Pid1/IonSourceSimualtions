"""
Solvers for electromagnetic potential and field problems.
"""

import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import spsolve
from scipy import interpolate


class PoissonSolverCylindrical:
    """
    Solves Poisson equation in cylindrical coordinates for ion source geometry.

    For a cylindrical tube of radius R and length L, we solve:
        1/r * d/dr(r * d phi/dr) + d²phi/dz² = -rho / epsilon_0

    where phi is the potential and rho is the charge density.

    This implementation uses finite differences with Dirichlet boundary conditions.
    """

    def __init__(self, radius, length, nr=50, nz=100):
        """
        Initialize the Poisson solver.

        Parameters
        ----------
        radius : float
            Tube radius [m]
        length : float
            Tube length [m]
        nr : int, optional
            Number of radial grid points
        nz : int, optional
            Number of axial grid points
        """
        self.radius = radius
        self.length = length
        self.nr = nr
        self.nz = nz

        # Create grid
        self.r = np.linspace(0, radius, nr)
        self.z = np.linspace(0, length, nz)
        self.dr = radius / (nr - 1)
        self.dz = length / (nz - 1)

        # Create 2D mesh
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")

    def _build_laplacian_matrix(self):
        """
        Build the discretized Laplacian operator matrix for cylindrical coordinates.

        Returns
        -------
        L : sparse matrix
            Laplacian operator in cylindrical coordinates
        """
        nr, nz = self.nr, self.nz
        N = nr * nz
        dr, dz = self.dr, self.dz

        # Radial operator coefficients (accounting for 1/r term)
        # d²/dr² + 1/r * d/dr
        r_diag = -2.0 / (dr**2) * np.ones(N)
        r_upper = np.zeros(N)
        r_lower = np.zeros(N)

        for i in range(nr):
            for j in range(nz):
                idx = i * nz + j
                if i > 0:
                    r_lower[idx] = 1.0 / (dr**2) + 1.0 / (2 * dr * self.r[i])
                if i < nr - 1:
                    r_upper[idx] = 1.0 / (dr**2) - 1.0 / (2 * dr * self.r[i])

        # Axial operator (standard d²/dz²)
        z_coeff = 1.0 / (dz**2)

        # Construct full matrix
        # Main diagonal
        diag_vals = [r_diag]

        # Off-diagonals for radial coupling
        diag_offs = [0]

        # Radial coupling (within same z-slice)
        r_upper_strip = np.zeros(N)
        r_lower_strip = np.zeros(N)
        for i in range(nr - 1):
            for j in range(nz):
                idx = i * nz + j
                next_idx = (i + 1) * nz + j
                r_upper_strip[idx] = 1.0 / (dr**2) - 1.0 / (2 * dr * self.r[i])
                r_lower_strip[next_idx] = 1.0 / (dr**2) + 1.0 / (2 * dr * self.r[i])

        diag_vals.append(r_upper_strip)
        diag_vals.append(r_lower_strip)
        diag_offs.extend([nz, -nz])

        # Axial coupling
        z_upper = z_coeff * np.ones(N)
        z_lower = z_coeff * np.ones(N)
        diag_vals.extend([z_upper, z_lower])
        diag_offs.extend([1, -1])

        # Apply boundary conditions on diagonals
        for i in range(nz):
            r_diag[i] -= r_upper_strip[i]  # r=0 boundary
            r_diag[-(i + 1)] -= r_lower_strip[-(i + 1)]  # r=R boundary
            r_diag[i] -= z_lower[i]  # z=0 boundary
            r_diag[-(nz + i)] -= z_upper[-(nz + i)]  # z=L boundary

        L = diags(diag_vals, diag_offs, shape=(N, N), format="csr")
        return L

    def solve(self, phi_boundary, charge_density=None):
        """
        Solve Poisson equation.

        Parameters
        ----------
        phi_boundary : dict
            Boundary conditions with keys: 'r0', 'rR', 'z0', 'zL'
            Each value is the potential [V] at that boundary
        charge_density : ndarray, optional
            Charge density distribution [C/m³], shape (nr, nz)
            If None, assumes zero charge

        Returns
        -------
        potential : ndarray
            Electric potential [V], shape (nr, nz)
        """
        from ..constants import epsilon_0

        nr, nz = self.nr, self.nz

        # Set up RHS
        if charge_density is None:
            rhs = np.zeros(nr * nz)
        else:
            rhs = -charge_density.flatten() / epsilon_0

        # Build Laplacian
        L = self._build_laplacian_matrix()

        # Apply boundary conditions
        # r = 0 (axis)
        phi_r0 = phi_boundary.get("r0", 0.0)

        # r = R (wall)
        phi_rR = phi_boundary.get("rR", 0.0)

        # z = 0 (entrance)
        phi_z0 = phi_boundary.get("z0", 0.0)

        # z = L (exit)
        phi_zL = phi_boundary.get("zL", 0.0)

        # Modify matrix and RHS for boundaries
        for j in range(nz):
            idx_r0 = j
            rhs[idx_r0] = phi_r0
            L[idx_r0, :] = 0
            L[idx_r0, idx_r0] = 1.0

            idx_rR = (nr - 1) * nz + j
            rhs[idx_rR] = phi_rR
            L[idx_rR, :] = 0
            L[idx_rR, idx_rR] = 1.0

        for i in range(nr):
            idx_z0 = i * nz
            rhs[idx_z0] = phi_z0
            L[idx_z0, :] = 0
            L[idx_z0, idx_z0] = 1.0

            idx_zL = i * nz + (nz - 1)
            rhs[idx_zL] = phi_zL
            L[idx_zL, :] = 0
            L[idx_zL, idx_zL] = 1.0

        # Solve
        phi_vec = spsolve(L, rhs)
        potential = phi_vec.reshape((nr, nz))

        return potential

    def get_electric_field(self, potential):
        """
        Calculate electric field from potential.

        E_r = -d(phi)/dr
        E_z = -d(phi)/dz

        Parameters
        ----------
        potential : ndarray
            Electric potential [V], shape (nr, nz)

        Returns
        -------
        E_r : ndarray
            Radial electric field [V/m], shape (nr, nz)
        E_z : ndarray
            Axial electric field [V/m], shape (nr, nz)
        """
        E_r = -np.gradient(potential, self.dr, axis=0)
        E_z = -np.gradient(potential, self.dz, axis=1)

        return E_r, E_z

    def get_field_at_point(self, r, z, potential):
        """
        Interpolate electric field at arbitrary point.

        Parameters
        ----------
        r : float
            Radial coordinate [m]
        z : float
            Axial coordinate [m]
        potential : ndarray
            Electric potential [V], shape (nr, nz)

        Returns
        -------
        E_r, E_z : float
            Electric field components [V/m]
        """
        E_r, E_z = self.get_electric_field(potential)

        f_Er = interpolate.RegularGridInterpolator(
            (self.r, self.z), E_r, bounds_error=False, fill_value=0
        )
        f_Ez = interpolate.RegularGridInterpolator(
            (self.r, self.z), E_z, bounds_error=False, fill_value=0
        )

        E_r_val = float(f_Er([r, z]))
        E_z_val = float(f_Ez([r, z]))

        return E_r_val, E_z_val
