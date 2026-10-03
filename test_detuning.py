import numpy as np
import matplotlib.pyplot as plt

from double_dot import (
    solve_double_dot,
    energies_to_meV,
    X,
    Y,
)


# ============================================================
# PARAMETERS
# ============================================================

V_right = -0.20
V_barrier = -0.30

# Sweep the left-dot control
V_left_values = np.linspace(-0.10, -0.30, 21)


# ============================================================
# STORAGE
# ============================================================

left_probability = []
right_probability = []
ground_energy = []


# ============================================================
# DETUNING SWEEP
# ============================================================

for V_left in V_left_values:

    _, energies, states = solve_double_dot(
        V_left=V_left,
        V_right=V_right,
        V_barrier=V_barrier,
        num_states=2,
    )

    E = energies_to_meV(energies)

    ground_energy.append(E[0])

    # Ground-state wavefunction
    psi0 = states[:, 0].reshape(X.shape)

    # Continuum-normalized probability density
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    probability = np.abs(psi0) ** 2 / (dx * dy)

    # --------------------------------------------------------
    # Integrate probability on each side of the device
    # --------------------------------------------------------

    left_mask = X < 0
    right_mask = X >= 0

    P_left = np.sum(
        probability[left_mask]
    ) * dx * dy

    P_right = np.sum(
        probability[right_mask]
    ) * dx * dy

    left_probability.append(P_left)
    right_probability.append(P_right)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nDetuning sweep")
print("----------------------------------------------")
print("V_left (V)    P_left      P_right      E0 (meV)")

for i in range(len(V_left_values)):

    print(
        f"{V_left_values[i]: .3f}"
        f"        {left_probability[i]:.4f}"
        f"       {right_probability[i]:.4f}"
        f"       {ground_energy[i]:.4f}"
    )


# ============================================================
# PLOT LOCALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    V_left_values,
    left_probability,
    marker="o",
    label="Left-dot probability",
)

plt.plot(
    V_left_values,
    right_probability,
    marker="o",
    label="Right-dot probability",
)

plt.xlabel("Left-dot control parameter (V)")
plt.ylabel("Ground-state probability")
plt.title("Double-dot ground-state localization")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "double_dot_localization.png",
    dpi=300,
    bbox_inches="tight",
)

print("\nSaved: double_dot_localization.png")
