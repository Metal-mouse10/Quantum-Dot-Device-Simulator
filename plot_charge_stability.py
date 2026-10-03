import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from charge_stability import find_ground_charge_state


# --------------------------------------------------
# Gate-voltage sweep
# --------------------------------------------------

V_left_values = np.linspace(-0.05, -0.50, 81)
V_right_values = np.linspace(-0.05, -0.50, 81)


N_left_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
)

N_right_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
)


# --------------------------------------------------
# Calculate ground-state charge configuration
# --------------------------------------------------

for i, V_right in enumerate(V_right_values):

    for j, V_left in enumerate(V_left_values):

        state, energy = find_ground_charge_state(
            V_left,
            V_right,
            max_electrons=5
        )

        N_left_map[i, j] = state[0]
        N_right_map[i, j] = state[1]


# Encode (N_left, N_right)
charge_map = 10 * N_left_map + N_right_map


# --------------------------------------------------
# Identify charge states actually present
# --------------------------------------------------

charge_states = sorted(np.unique(charge_map))

print("\nCharge states present:")
print("---------------------")

for state_code in charge_states:

    N_left = state_code // 10
    N_right = state_code % 10

    print(
        f"({N_left}, {N_right})"
    )


# --------------------------------------------------
# Discrete boundaries
# --------------------------------------------------

bounds = np.arange(
    charge_map.min() - 5,
    charge_map.max() + 6,
    10
)

norm = BoundaryNorm(
    bounds,
    ncolors=256
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

image = plt.imshow(
    charge_map,
    origin="lower",
    extent=[
        V_left_values[0],
        V_left_values[-1],
        V_right_values[0],
        V_right_values[-1]
    ],
    aspect="auto",
    interpolation="nearest",
    cmap="viridis"
)


plt.xlabel("Left-dot gate control (V)")
plt.ylabel("Right-dot gate control (V)")

plt.title(
    "Double Quantum Dot Charge Stability Diagram"
)


# --------------------------------------------------
# Add charge-state labels
# --------------------------------------------------

for state_code in charge_states:

    mask = charge_map == state_code

    if np.any(mask):

        rows, cols = np.where(mask)

        row_center = int(np.mean(rows))
        col_center = int(np.mean(cols))

        V_left_center = V_left_values[col_center]
        V_right_center = V_right_values[row_center]

        N_left = state_code // 10
        N_right = state_code % 10

        plt.text(
            V_left_center,
            V_right_center,
            f"({N_left},{N_right})",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold"
        )


plt.tight_layout()

plt.savefig(
    "charge_stability_diagram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: charge_stability_diagram.png")
