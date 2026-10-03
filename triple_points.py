import numpy as np

from charge_stability import (
    E_C_LEFT,
    E_C_RIGHT,
    E_M,
    ALPHA_LL,
    ALPHA_LR,
    ALPHA_RL,
    ALPHA_RR,
    charge_energy,
    induced_charge,
)


# ============================================================
# TRANSITION ENERGIES
# ============================================================

def left_transition_energy(
    N_left,
    N_right,
    V_left,
    V_right,
):
    """
    Energy required to add one electron to the left dot.

    ΔE_L =
        E(N_L+1,N_R) - E(N_L,N_R)
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
    Energy required to add one electron to the right dot.

    ΔE_R =
        E(N_L,N_R+1) - E(N_L,N_R)
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
# ANALYTIC TRIPLE POINT
# ============================================================

def find_triple_point(
    N_left,
    N_right,
):
    """
    Calculate the triple point analytically.

    The triple point satisfies:

        ΔE_L = 0
        ΔE_R = 0

    First solve for the induced charges
    (n_g_left, n_g_right), then convert them
    to gate-control values.
    """

    # --------------------------------------------------------
    # Solve for induced charges
    # --------------------------------------------------------

    matrix = np.array([
        [E_C_LEFT, E_M],
        [E_M, E_C_RIGHT],
    ])

    rhs = np.array([
        E_C_LEFT * (N_left + 0.5)
        + E_M * N_right,

        E_M * N_left
        + E_C_RIGHT * (N_right + 0.5),
    ])

    n_g_left, n_g_right = np.linalg.solve(
        matrix,
        rhs,
    )


    # --------------------------------------------------------
    # Convert induced charge to gate controls
    #
    # n_g_left =
    #   ALPHA_LL*(-V_left)
    #   + ALPHA_LR*(-V_right)
    #
    # n_g_right =
    #   ALPHA_RL*(-V_left)
    #   + ALPHA_RR*(-V_right)
    # --------------------------------------------------------

    gate_matrix = np.array([
        [ALPHA_LL, ALPHA_LR],
        [ALPHA_RL, ALPHA_RR],
    ])

    gate_vector = np.linalg.solve(
        gate_matrix,
        np.array([
            n_g_left,
            n_g_right,
        ]),
    )

    V_left = -gate_vector[0]
    V_right = -gate_vector[1]


    return (
        V_left,
        V_right,
        n_g_left,
        n_g_right,
    )


# ============================================================
# VERIFY TRIPLE POINT
# ============================================================

if __name__ == "__main__":

    N_left = 1
    N_right = 1

    (
        V_left,
        V_right,
        n_g_left,
        n_g_right,
    ) = find_triple_point(
        N_left,
        N_right,
    )


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


    print()
    print("Double Quantum Dot Triple Point")
    print("--------------------------------")

    print(
        f"Reference charge state: "
        f"({N_left}, {N_right})"
    )

    print()

    print(
        f"Induced charge:"
    )

    print(
        f"n_g_left  = {n_g_left:.8f}"
    )

    print(
        f"n_g_right = {n_g_right:.8f}"
    )

    print()

    print(
        f"Triple-point gate controls:"
    )

    print(
        f"V_left  = {V_left:.8f} V"
    )

    print(
        f"V_right = {V_right:.8f} V"
    )

    print()

    print(
        f"Left transition ΔE  = "
        f"{delta_left:.6e} meV"
    )

    print(
        f"Right transition ΔE = "
        f"{delta_right:.6e} meV"
    )

    print()

    print(
        f"Reference-state energy = "
        f"{charge_energy("
        f"N_left, N_right, "
        f"V_left, V_right"
        f"):.6e} meV"
    )
