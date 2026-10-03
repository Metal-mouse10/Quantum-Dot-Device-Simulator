import matplotlib.pyplot as plt
import numpy as np

from double_dot import (
    solve_double_dot,
    energies_to_meV,
    X,
    Y,
)


# Solve symmetric double dot
U_meV, energies, states = solve_double_dot(
    V_left=-0.20,
    V_right=-0.20,
    V_barrier=-0.30,
    num_states=4,
)

energies_meV = energies_to_meV(energies)

dx = X[0, 1] - X[0, 0]
dy = Y[1, 0] - Y[0, 0]


# ------------------------------------------------------------
# Ground state
# ------------------------------------------------------------

psi0 = states[:, 0].reshape(X.shape)

prob0 = np.abs(psi0)**2 / (dx * dy)


# ------------------------------------------------------------
# First excited state
# ------------------------------------------------------------

psi1 = states[:, 1].reshape(X.shape)

prob1 = np.abs(psi1)**2 / (dx * dy)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# Ground state
im0 = axes[0].pcolormesh(
    X * 1e9,
    Y * 1e9,
    prob0,
    shading="auto",
)

axes[0].set_title(
    f"Ground state |ψ₀|²\nE₀ = {energies_meV[0]:.4f} meV"
)

axes[0].set_xlabel("x (nm)")
axes[0].set_ylabel("y (nm)")

plt.colorbar(
    im0,
    ax=axes[0],
    label=r"Probability density (m$^{-2}$)"
)


# First excited state
im1 = axes[1].pcolormesh(
    X * 1e9,
    Y * 1e9,
    prob1,
    shading="auto",
)

axes[1].set_title(
    f"First excited state |ψ₁|²\nE₁ = {energies_meV[1]:.4f} meV"
)

axes[1].set_xlabel("x (nm)")
axes[1].set_ylabel("y (nm)")

plt.colorbar(
    im1,
    ax=axes[1],
    label=r"Probability density (m$^{-2}$)"
)


plt.tight_layout()
plt.show()
