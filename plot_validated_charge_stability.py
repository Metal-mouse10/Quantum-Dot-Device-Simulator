import numpy as np
import matplotlib.pyplot as plt

from charge_stability import find_ground_charge_state
from validate_triple_points import validate_triple_point


# ============================================================
# GATE-VOLTAGE RANGE
# ============================================================

V_left_values = np.linspace(-0.05, -0.50, 201)
V_right_values = np.linspace(-0.05, -0.50, 201)


# ============================================================
# CALCULATE CHARGE-STABILITY MAP
# ============================================================

N_left_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int,
)

N_right_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int,
)


for i, V_right in enumerate(V_right_values):

    for j, V_left in enumerate(V_left_values):

        state, energy = find_ground_charge_state(
            V_left=V_left,
            V_right=V_right,
            max_electrons=6,
        )

        N_left_map[i, j] = state[0]
        N_right_map[i, j] = state[1]


# Encode charge configuration
charge_map = (
    10 * N_left_map
    + N_right_map
)

unique_states = sorted(
    np.unique(charge_map)
)


# ============================================================
# VALIDATE TRIPLE POINTS
# ============================================================

candidate_states = [
    (0, 0),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
]


validated_triple_points = []


for N_left, N_right in candidate_states:

    (
        V_left,
        V_right,
        energies,
        energy_spread,
        is_valid,
    ) = validate_triple_point(
        N_left,
        N_right,
    )

    # Only retain validated ground-state triple points
    if not is_valid:
        continue

    # Only retain points inside plotted region
    if not (
        V_left_values.min()
        <= V_left
        <= V_left_values.max()
    ):
        continue

    if not (
        V_right_values.min()
        <= V_right
        <= V_right_values.max()
    ):
        continue

    validated_triple_points.append(
        {
            "N_left": N_left,
            "N_right": N_right,
            "V_left": V_left,
            "V_right": V_right,
            "energy_spread": energy_spread,
        }
    )


# ============================================================
# PRINT VALIDATED TRIPLE POINTS
# ============================================================

print()
print("Validated ground-state triple points")
print("====================================")

for point in validated_triple_points:

    print(
        f"({point['N_left']},{point['N_right']})"
        f"  "
        f"V_left = {point['V_left']:.6f} V"
        f"  "
        f"V_right = {point['V_right']:.6f} V"
    )


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


image = ax.imshow(
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


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    r"Left-dot gate control $V_L$ (V)",
    fontsize=12,
)

ax.set_ylabel(
    r"Right-dot gate control $V_R$ (V)",
    fontsize=12,
)

ax.set_title(
    "Double Quantum Dot Charge Stability",
    fontsize=14,
)


# ============================================================
# CHARGE-REGION LABELS
# ============================================================

for state_code in unique_states:

    mask = charge_map == state_code

    rows, cols = np.where(mask)

    if len(rows) == 0:
        continue

    row_center = int(
        np.mean(rows)
    )

    col_center = int(
        np.mean(cols)
    )

    V_left_center = (
        V_left_values[col_center]
    )

    V_right_center = (
        V_right_values[row_center]
    )

    N_left = state_code // 10
    N_right = state_code % 10

    ax.text(
        V_left_center,
        V_right_center,
        f"({N_left},{N_right})",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold",
    )


# ============================================================
# TRIPLE-POINT MARKERS
# ============================================================

for point in validated_triple_points:

    V_left = point["V_left"]
    V_right = point["V_right"]

    N_left = point["N_left"]
    N_right = point["N_right"]

    ax.scatter(
        V_left,
        V_right,
        s=90,
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=5,
    )

    ax.annotate(
        f"TP ({N_left},{N_right})",
        xy=(V_left, V_right),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
        color="white",
        fontweight="bold",
        zorder=6,
    )


# ============================================================
# COLORBAR
# ============================================================

colorbar = fig.colorbar(
    image,
    ax=ax,
)

colorbar.set_label(
    r"Charge configuration $(N_L,N_R)$",
    fontsize=11,
)


# ============================================================
# SAVE
# ============================================================

fig.tight_layout()

fig.savefig(
    "validated_charge_stability.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print(
    "Saved: validated_charge_stability.png"
)
