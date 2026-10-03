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

# Effective conversion from gate-control parameter to
# dimensionless induced charge.

ALPHA_LEFT = 5.0
ALPHA_RIGHT = 5.0


# ============================================================
# GATE-INDUCED CHARGE
# ============================================================

def induced_charge(V_left, V_right):
    """
    Convert effective gate-control parameters into
    dimensionless induced charges.

    These are phenomenological parameters.

    Returns
    -------
    n_g_left : float
    n_g_right : float
    """

    n_g_left = ALPHA_LEFT * (-V_left)
    n_g_right = ALPHA_RIGHT * (-V_right)

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
    max_electrons=5,
):
    """
    Find the lowest-energy charge configuration.
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

            if energy < lowest_energy:

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
        max_electrons=5,
    )

    print("\nDouble Quantum Dot Charge Model")
    print("--------------------------------")

    print(
        f"Gate controls: "
        f"V_left = {V_left:.2f} V, "
        f"V_right = {V_right:.2f} V"
    )

    print(
        f"Induced charges: "
        f"n_g_left = {induced_charge(V_left, V_right)[0]:.3f}, "
        f"n_g_right = {induced_charge(V_left, V_right)[1]:.3f}"
    )

    print(
        f"Ground charge state: "
        f"(N_left, N_right) = {state}"
    )

    print(
        f"Energy = {energy:.6f} meV"
    )
