import matplotlib.pyplot as plt

from electrostatics import (
    X,
    Y,
    gate_potential,
    electron_potential_energy,
)


# ============================================================
# PARAMETERS
# ============================================================

V_left = -0.30
V_plunger = 0.10
V_right = -0.30


# ============================================================
# CALCULATE
# ============================================================

phi = gate_potential(
    V_left=V_left,
    V_plunger=V_plunger,
    V_right=V_right,
)

U_meV = electron_potential_energy(
    V_left=V_left,
    V_plunger=V_plunger,
    V_right=V_right,
)


# ============================================================
# CONVERT COORDINATES TO nm
# ============================================================

X_nm = X * 1e9
Y_nm = Y * 1e9


# ============================================================
# ELECTROSTATIC POTENTIAL
# ============================================================

plt.figure(figsize=(7, 6))

plt.pcolormesh(
    X_nm,
    Y_nm,
    phi,
    shading="auto",
)

plt.colorbar(label="Effective electrostatic potential (V)")

plt.xlabel("x (nm)")
plt.ylabel("y (nm)")

plt.title("Geometry-Aware Effective Electrostatic Potential")

plt.tight_layout()

plt.savefig(
    "electrostatic_potential.png",
    dpi=300,
)

plt.show()


# ============================================================
# ELECTRON POTENTIAL ENERGY
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

plt.title("Electron Potential Energy")

plt.tight_layout()

plt.savefig(
    "electron_potential_energy.png",
    dpi=300,
)

plt.show()

# ============================================================
# SHIFTED ELECTRON POTENTIAL ENERGY
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
    "Finite-Depth Electrostatic Confinement Potential"
)

plt.tight_layout()

plt.savefig(
    "electron_potential_energy.png",
    dpi=300,
)

plt.show()
