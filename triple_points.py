import numpy as np

from charge_stability import charge_energy


# ============================================================
# CHARGE TRANSITION FUNCTIONS
# ============================================================

def left_transition_energy(
    N_left,
    N_right,
    V_left,
    V_right,
):
    """
    Energy difference for adding one electron
    to the left dot.

    ΔE_L = E(N_L+1,N_R) - E(N_L,N_R)
    """

    return (
        charge_energy(
            N_left + 1,
            N_right,
            V_left,
            V_right,
        )
        -
        charge_energy(
            N_left,
            N_right,
            V_left,
            V_right,
        )
    )


def right_transition_energy(
    N_left,
    N_right,
    V_left,
    V_right,
):
    """
    Energy difference for adding one electron
    to the right dot.

    ΔE_R = E(N_L,N_R+1) - E(N_L,N_R)
    """

    return (
        charge_energy(
            N_left,
            N_right + 1,
            V_left,
            V_right,
        )
        -
        charge_energy(
            N_left,
            N_right,
            V_left,
            V_right,
        )
    )


# ============================================================
# NUMERICAL SEARCH FOR A TRIPLE POINT
# ============================================================

def find_triple_point(
    N_left,
    N_right,
    V_left_range=(-0.50, -0.05),
    V_right_range=(-0.50, -0.05),
    resolution=201,
):
    """
    Find the approximate gate-voltage location where
    both left and right charge transitions occur.

    This searches for the point minimizing:

        ΔE_L^2 + ΔE_R^2
    """

    V_left_values = np.linspace(
        V_left_range[0],
        V_left_range[1],
        resolution,
    )

    V_right_values = np.linspace(
        V_right_range[0],
        V_right_range[1],
        resolution,
    )

    best_error = np.inf
    best_point = None

    for V_left in V_left_values:

        for V_right in V_right_values:

            delta_left = left_transition_energy(
                N_left,
                N_right,
                V_left,
                V_right,
            )

            delta_right = right_transition_energy(
                N_left,
                N_right,
                V_left,
                V_right,
            )

            error = (
                delta_left**2
                + delta_right**2
            )

            if error < best_error:

                best_error = error

                best_point = (
                    V_left,
                    V_right,
                    delta_left,
                    delta_right,
                )

    return best_point, best_error


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    N_left = 1
    N_right = 1

    point, error = find_triple_point(
        N_left,
        N_right,
    )

    V_left, V_right, delta_left, delta_right = point

    print()
    print("Double Quantum Dot Triple-Point Search")
    print("----------------------------------------")

    print(
        f"Reference charge state: "
        f"({N_left}, {N_right})"
    )

    print(
        f"Approximate triple point:"
    )

    print(
        f"V_left  = {V_left:.6f} V"
    )

    print(
        f"V_right = {V_right:.6f} V"
    )

    print()
    print(
        f"Left transition ΔE = "
        f"{delta_left:.6e} meV"
    )

    print(
        f"Right transition ΔE = "
        f"{delta_right:.6e} meV"
    )

    print(
        f"Search error = "
        f"{error:.6e}"
    )
