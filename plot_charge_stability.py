import numpy as np
import matplotlib.pyplot as plt

from charge_stability import find_ground_charge_state


# Gate-voltage sweep
V_left_values = np.linspace(-0.05, -0.50, 61)
V_right_values = np.linspace(-0.05, -0.50, 61)


# Store ground-state charge configuration
N_left_map = np.zeros(
    (len(V_right_values), len(V_left_values))
)

N_right_map = np.zeros(
    (len(V_right_values), len(V_left_values))
)


# Sweep gate voltages
for i, V_right in enumerate(V_right_values):

    for j, V_left in enumerate(V_left_values):

        state, energy = find_ground_charge_state(
            V_left,
            V_right,
            max_electrons=5
        )

        N_left_map[i, j] = state[0]
        N_right_map[i, j] = state[1]


# Encode charge configuration as
# 10 * N_left + N_right
charge_map = 10 * N_left_map + N_right_map


# Plot
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
    aspect="auto"
)

plt.xlabel("Left-dot gate control (V)")
plt.ylabel("Right-dot gate control (V)")
plt.title("Double Quantum Dot Charge Stability Diagram")

cbar = plt.colorbar(image)
cbar.set_label(r"Charge configuration $(N_L,N_R)$")


plt.tight_layout()

plt.savefig(
    "charge_stability_diagram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Saved: charge_stability_diagram.png")
