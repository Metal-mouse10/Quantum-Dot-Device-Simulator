import numpy as np

from double_dot import (
    solve_double_dot,
    energies_to_meV,
)


# ============================================================
# BARRIER SWEEP
# ============================================================

barriers = [
    -0.20,
    -0.30,
    -0.40,
    -0.50,
]


print("\nBarrier sweep")
print("----------------------------------------------")
print("Barrier (V)    E0 (meV)    E1 (meV)    Delta E (meV)")


for barrier in barriers:

    # Solve double-dot system
    _, energies, _ = solve_double_dot(
        V_left=-0.20,
        V_right=-0.20,
        V_barrier=barrier,
        num_states=4,
    )

    # Convert joules → meV
    E = energies_to_meV(energies)

    # Lowest-state splitting
    delta_E = E[1] - E[0]

    print(
        f"{barrier: .2f}"
        f"        {E[0]: .4f}"
        f"       {E[1]: .4f}"
        f"       {delta_E: .4f}"
    )
