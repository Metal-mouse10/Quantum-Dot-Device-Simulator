import numpy as np


# ============================================================
# ENERGY SCALES
# ============================================================

# Effective charging energies.
# These are model parameters, not experimentally calibrated.

E_C_LEFT = 5.0       # meV
E_C_RIGHT = 5.0      # meV

# Mutual capacitive coupling energy
E_M = 0.8             # meV


# ============================================================
# GATE COUPLING
# ============================================================

# Direct gate-to-dot coupling
ALPHA_LL = 5.0        # Left gate -> Left dot
ALPHA_RR = 5.0        # Right gate -> Right dot

# Cross-capacitance
# A gate also weakly influences the opposite dot.
ALPHA_LR = 0.8        # Right gate -> Left dot
ALPHA_RL = 0.8        # Left gate -> Right dot


# ============================================================
# GATE-INDUCED CHARGE
# ============================================================

def induced_charge(V_left, V_right):
    """
    Calculate dimensionless gate-induced charges.

    The model includes both direct and cross-capacitance:

        n_g_left =
            ALPHA_LL * (-V_left)
            + ALPHA_LR * (-V_right)

        n_g_right =
            ALPHA_RR * (-V_right)
            + ALPHA_RL * (-V_left)

    These coupling parameters are phenomenological and are
    not experimentally calibrated.
    """

    n_g_left = (
        ALPHA_LL * (-V_left)
        + ALPHA_LR * (-V_right)
    )

    n_g_right = (
        ALPHA_RR * (-V_right)
        + ALPHA_RL * (-V_left)
    )

    return n_g_left, n_g_right


# ============================================================
# CHARGE CONFIGURATION ENERGY
# ============================================================

def charge_energy(
    N_left,
    N_right,
    V_left,
    V_right,
):
    """
    Calculate the constant-interaction energy of a
    double-dot charge configuration.

    Energy returned in meV.
    """

    n_g_left, n_g_right = induced_charge(
        V_left,
        V_right,
    )

    delta_left = N_left - n_g_left
    delta_right = N_right - n_g_right

    energy = (
        0.5 * E_C_LEFT * delta_left**2
        + 0.5 * E_C_RIGHT * delta_right**2
        + E_M * delta_left * delta_right
    )

    return energy


# ============================================================
# FIND GROUND CHARGE STATE
# ============================================================

def find_ground_charge_state(
    V_left,
    V_right,
    max_electrons=6,
):
    """
    Find the lowest-energy charge configuration.

    If multiple configurations are exactly degenerate
    within numerical tolerance, the first one encountered
    is returned.
    """

    lowest_energy = np.inf
    ground_state = None

    for N_left in range(max_electrons + 1):

        for N_right in range(max_electrons + 1):

            energy = charge_energy(
                N_left=N_left,
                N_right=N_right,
                V_left=V_left,
                V_right=V_right,
            )

            if energy < lowest_energy - 1e-12:

                lowest_energy = energy

                ground_state = (
                    N_left,
                    N_right,
                )

    return ground_state, lowest_energy


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    V_left = -0.20
    V_right = -0.20

    state, energy = find_ground_charge_state(
        V_left=V_left,
        V_right=V_right,
        max_electrons=6,
    )

    n_g_left, n_g_right = induced_charge(
        V_left,
        V_right,
    )

    print()
    print("Double Quantum Dot Charge Model")
    print("--------------------------------")

    print(
        f"Gate controls: "
        f"V_left = {V_left:.2f} V, "
        f"V_right = {V_right:.2f} V"
    )

    print(
        f"Induced charges: "
        f"n_g_left = {n_g_left:.3f}, "
        f"n_g_right = {n_g_right:.3f}"
    )

    print(
        f"Ground charge state: "
        f"(N_left, N_right) = {state}"
    )

    print(
        f"Energy = {energy:.6f} meV"
    )
