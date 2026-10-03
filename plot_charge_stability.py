import numpy as np
import matplotlib.pyplot as plt

from charge_stability import find_ground_charge_state


# ============================================================
# GATE-VOLTAGE RANGE
# ============================================================

V_left_values = np.linspace(-0.05, -0.50, 101)
V_right_values = np.linspace(-0.05, -0.50, 101)


# ============================================================
# STORAGE
# ============================================================

N_left_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
)

N_right_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
)


# ============================================================
# CHARGE-STABILITY CALCULATION
# ============================================================

for i, V_right in enumerate(V_right_values):

    for j, V_left in enumerate(V_left_values):

        state, energy = find_ground_charge_state(
            V_left=V_left,
            V_right=V_right,
            max_electrons=6,
        )

        N_left_map[i, j] = state[0]
        N_right_map[i, j] = state[1]


# ============================================================
# ENCODE CHARGE CONFIGURATION
# ============================================================

charge_map = 10 * N_left_map + N_right_map


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 7))

unique_states = sorted(np.unique(charge_map))

image = plt.imshow(
    charge_map,
    origin="lower",
    extent=[
        V_left_values[0],
        V_left_values[-1],
        V_right_values[0],
        V_right_values[-1],
    ],
    aspect="equal",
    interpolation="nearest",
    cmap="viridis",
)


plt.xlabel("Left-dot gate control (V)")
plt.ylabel("Right-dot gate control (V)")

plt.title(
    "Double Quantum Dot Charge Stability Diagram"
)


# ============================================================
# LABEL CHARGE REGIONS
# ============================================================

for state_code in unique_states:

    mask = charge_map == state_code

    rows, cols = np.where(mask)

    if len(rows) == 0:
        continue

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
        fontweight="bold",
    )


# ============================================================
# REPORT STATES
# ============================================================

print()
print("Charge configurations present")
print("--------------------------------")

for state_code in unique_states:

    N_left = state_code // 10
    N_right = state_code % 10

    print(
        f"({N_left}, {N_right})"
    )


plt.tight_layout()

plt.savefig(
    "charge_stability_diagram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Saved: charge_stability_diagram.png")
