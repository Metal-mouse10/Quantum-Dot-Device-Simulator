import numpy as np


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

e = 1.602176634e-19


# ============================================================
# DOUBLE-DOT CAPACITANCE MODEL
# ============================================================

# Effective capacitances
#
# These are model parameters, not experimentally calibrated
# device values.

C_L = 10e-18       # Left dot total capacitance
C_R = 10e-18       # Right dot total capacitance
C_m = 2e-18        # Mutual capacitance


# ============================================================
# CHARGING ENERGY
# ============================================================

def charging_energy(
    N_left,
    N_right,
    V_left,
    V_right,
):
    """
    Calculate the electrostatic energy of a double-dot
    charge configuration using a constant-interaction model.

    Parameters
    ----------
    N_left : int
        Number of excess electrons on the left dot.

    N_right : int
        Number of excess electrons on the right dot.

    V_left : float
        Left gate control parameter in volts.

    V_right : float
        Right gate control parameter in volts.

    Returns
    -------
    energy : float
        Electrostatic energy in joules.

    Notes
    -----
    This is a simplified constant-interaction model.

    The capacitances and gate couplings are effective model
    parameters and are not experimentally calibrated.
    """

    # --------------------------------------------------------
    # Gate-induced charge
    # --------------------------------------------------------

    Q_left = C_L * V_left
    Q_right = C_R * V_right

    # Convert electron numbers into charge
    q_left = -N_left * e
    q_right = -N_right * e

    # Effective charge relative to gate-induced charge
    delta_Q_left = q_left - Q_left
    delta_Q_right = q_right - Q_right

    # --------------------------------------------------------
    # Capacitance matrix
    # --------------------------------------------------------

    C_matrix = np.array([
        [C_L + C_m, -C_m],
        [-C_m, C_R + C_m]
    ])

    # Inverse capacitance matrix
    C_inverse = np.linalg.inv(C_matrix)

    # Charge vector
    delta_Q = np.array([
        delta_Q_left,
        delta_Q_right
    ])

    # --------------------------------------------------------
    # Electrostatic energy
    #
    # E = 1/2 Q^T C^-1 Q
    # --------------------------------------------------------

    energy = 0.5 * delta_Q @ C_inverse @ delta_Q

    return energy


# ============================================================
# CONVERT JOULES → meV
# ============================================================

def joule_to_meV(energy):
    """
    Convert energy from joules to meV.
    """

    return energy / e * 1000.0


# ============================================================
# FIND LOWEST-ENERGY CHARGE CONFIGURATION
# ============================================================

def find_ground_charge_state(
    V_left,
    V_right,
    max_electrons=3,
):
    """
    Find the lowest-energy charge configuration.

    Searches:

        N_left  = 0 ... max_electrons
        N_right = 0 ... max_electrons

    Returns
    -------
    ground_state : tuple
        (N_left, N_right)

    ground_energy : float
        Ground-state energy in joules.
    """

    lowest_energy = np.inf
    ground_state = None

    for N_left in range(max_electrons + 1):

        for N_right in range(max_electrons + 1):

            energy = charging_energy(
                N_left=N_left,
                N_right=N_right,
                V_left=V_left,
                V_right=V_right,
            )

            if energy < lowest_energy:

                lowest_energy = energy

                ground_state = (
                    N_left,
                    N_right
                )

    return ground_state, lowest_energy


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    V_left = -0.20
    V_right = -0.20

    state, energy = find_ground_charge_state(
        V_left=V_left,
        V_right=V_right,
        max_electrons=3,
    )

    print("\nDouble Quantum Dot Charge Model")
    print("--------------------------------")

    print(
        f"Gate controls:"
        f" V_left = {V_left:.2f} V,"
        f" V_right = {V_right:.2f} V"
    )

    print(
        f"Ground charge state: "
        f"(N_left, N_right) = {state}"
    )

    print(
        f"Energy = "
        f"{joule_to_meV(energy):.6f} meV"
    )
