import numpy as np
import matplotlib.pyplot as plt

from charge_stability import find_ground_charge_state
from triple_points import find_triple_point


# ============================================================
# GATE-VOLTAGE RANGE
# ============================================================

V_left_values = np.linspace(-0.05, -0.50, 101)
V_right_values = np.linspace(-0.05, -0.50, 101)


# ============================================================
# CALCULATE CHARGE-STABILITY MAP
# ============================================================

N_left_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
)

N_right_map = np.zeros(
    (len(V_right_values), len(V_left_values)),
    dtype=int
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
charge_map = 10 * N_left_map + N_right_map

unique_states = sorted(np.unique(charge_map))


# ============================================================
# CALCULATE TRIPLE POINTS
# ============================================================

triple_points = []


# Search charge configurations relevant to our map
for N_left in range(0, 4):

    for N_right in range(0, 4):

        try:

            (
                V_left,
                V_right,
                n_g_left,
                n_g_right,
            ) = find_triple_point(
                N_left,
                N_right,
            )

            # Only keep triple points inside plotted region
            if (
                V_left_values.min()
                <= V_left
                <= V_left_values.max()
                and
                V_right_values.min()
                <= V_right
                <= V_right_values.max()
            ):

                triple_points.append(
                    (
                        N_left,
                        N_right,
                        V_left,
                        V_right,
                    )
                )

        except np.linalg.LinAlgError:

            # Skip singular capacitance matrices
            continue


# ============================================================
# PRINT TRIPLE POINTS
# ============================================================

print()
print("Triple points inside plotted region")
print("-----------------------------------")

for (
    N_left,
    N_right,
    V_left,
    V_right,
) in triple_points:

    print(
        f"({N_left},{N_right}) "
        f"at "
        f"V_left = {V_left:.6f} V, "
        f"V_right = {V_right:.6f} V"
    )


# ============================================================
# PLOT CHARGE STABILITY DIAGRAM
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(
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
        fontsize=8,
        color="white",
        fontweight="bold",
    )


# ============================================================
# PLOT TRIPLE POINTS
# ============================================================

for (
    N_left,
    N_right,
    V_left,
    V_right,
) in triple_points:

    plt.scatter(
        V_left,
        V_right,
        s=70,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
        zorder=5,
    )

    plt.annotate(
        f"TP ({N_left},{N_right})",
        xy=(V_left, V_right),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color="white",
        fontweight="bold",
    )


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    "charge_stability_with_triple_points.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print(
    "Saved: "
    "charge_stability_with_triple_points.png"
)
