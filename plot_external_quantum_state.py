import numpy as np
import matplotlib.pyplot as plt

from electrostatics import (
    X,
    Y,
    electron_potential_energy,
)

from quantum_dot import (
    solve_external_potential,
)


# ============================================================
# GENERATE EXTERNAL ELECTROSTATIC POTENTIAL
# ============================================================

U_meV = electron_potential_energy(
    V_left=-0.30,
    V_plunger=0.10,
    V_right=-0.30,
)


# ============================================================
# SOLVE SCHRÖDINGER EQUATION
# ============================================================

energies, states = solve_external_potential(
    U_meV=U_meV,
    x_grid=X[0, :],
    y_grid=Y[:, 0],
    num_states=4,
)


# ============================================================
# CONVERT ENERGIES TO meV
# ============================================================

eV = 1.602176634e-19

energies_meV = (
    energies / eV * 1000.0
)

print("\nQuantum states")
print("--------------")

for i, energy in enumerate(energies_meV):
    print(f"E{i} = {energy:.6f} meV")


# ============================================================
# GRID SPACING
# ============================================================

dx = X[0, 1] - X[0, 0]
dy = Y[1, 0] - Y[0, 0]


# ============================================================
# RESHAPE WAVEFUNCTIONS
# ============================================================

psi0 = states[:, 0].reshape(X.shape)
psi1 = states[:, 1].reshape(X.shape)


# ============================================================
# PROBABILITY DENSITIES
# ============================================================

probability0 = (
    np.abs(psi0) ** 2
    / (dx * dy)
)

probability1 = (
    np.abs(psi1) ** 2
    / (dx * dy)
)


# ============================================================
# CONVERT COORDINATES TO nm
# ============================================================

X_nm = X * 1e9
Y_nm = Y * 1e9


# ============================================================
# PLOT 1 — ELECTROSTATIC ELECTRON POTENTIAL
# ============================================================

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    X_nm,
    Y_nm,
    U_meV,
    shading="auto",
)

plt.colorbar(
    label="Electron potential energy (meV)"
)

plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

plt.title(
    "Electrostatic Electron Potential"
)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2 — GROUND-STATE PROBABILITY DENSITY
# ============================================================

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    X_nm,
    Y_nm,
    probability0,
    shading="auto",
)

plt.colorbar(
    label="Probability density (m$^{-2}$)"
)

plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

plt.title(
    f"Ground State Probability Density\n"
    f"E₀ = {energies_meV[0]:.6f} meV"
)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3 — FIRST EXCITED-STATE PROBABILITY DENSITY
# ============================================================

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    X_nm,
    Y_nm,
    probability1,
    shading="auto",
)

plt.colorbar(
    label="Probability density (m$^{-2}$)"
)

plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

plt.title(
    f"First Excited State Probability Density\n"
    f"E₁ = {energies_meV[1]:.6f} meV"
)

plt.tight_layout()

plt.show()
