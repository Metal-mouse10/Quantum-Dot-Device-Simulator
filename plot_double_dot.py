import matplotlib.pyplot as plt
import numpy as np

from double_dot import (
    solve_double_dot,
    energies_to_meV,
    X,
    Y,
)


# ============================================================
# SOLVE DOUBLE DOT
# ============================================================

U_meV, energies, states = solve_double_dot(
    V_left=-0.20,
    V_right=-0.20,
    V_barrier=-0.30,
    num_states=4,
)

energies_meV = energies_to_meV(energies)


# ============================================================
# RESHAPE WAVEFUNCTIONS
# ============================================================

psi0 = states[:, 0].reshape(X.shape)
psi1 = states[:, 1].reshape(X.shape)


# ============================================================
# PLOT ACTUAL WAVEFUNCTIONS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# ------------------------------------------------------------
# Ground state
# ------------------------------------------------------------

vmax0 = np.max(np.abs(psi0))

im0 = axes[0].pcolormesh(
    X * 1e9,
    Y * 1e9,
    psi0,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax0,
    vmax=vmax0,
)

axes[0].set_title(
    f"Ground state ψ₀\nE₀ = {energies_meV[0]:.4f} meV"
)

axes[0].set_xlabel("x (nm)")
axes[0].set_ylabel("y (nm)")

plt.colorbar(
    im0,
    ax=axes[0],
    label=r"$\psi_0$"
)


# ------------------------------------------------------------
# First excited state
# ------------------------------------------------------------

vmax1 = np.max(np.abs(psi1))

im1 = axes[1].pcolormesh(
    X * 1e9,
    Y * 1e9,
    psi1,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax1,
    vmax=vmax1,
)

axes[1].set_title(
    f"First excited state ψ₁\nE₁ = {energies_meV[1]:.4f} meV"
)

axes[1].set_xlabel("x (nm)")
axes[1].set_ylabel("y (nm)")

plt.colorbar(
    im1,
    ax=axes[1],
    label=r"$\psi_1$"
)


plt.tight_layout()

plt.savefig(
    "double_dot_wavefunctions_signed.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved: double_dot_wavefunctions_signed.png")
