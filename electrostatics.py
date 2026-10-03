import numpy as np


# ============================================================
# DEVICE GRID
# ============================================================

Lx = 100e-9
Ly = 100e-9

Nx = 201
Ny = 201

x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)

X, Y = np.meshgrid(x, y)


# ============================================================
# EFFECTIVE ELECTROSTATIC GATE MODEL
# ============================================================

def gate_potential(
    V_left=-0.30,
    V_plunger=0.10,
    V_right=-0.30,
    gate_separation_nm=25.0,
    gate_depth_nm=20.0,
    softening_nm=2.0,
):
    """
    Calculate an effective electrostatic potential at the
    semiconductor electron plane produced by three gates.

    The gates are represented as effective localized sources.
    A finite vertical gate-to-electron-plane distance is
    included through a softened 3D distance.

    This is an effective electrostatic approximation.
    It is NOT a full Poisson-equation solution.
    """

    # --------------------------------------------------------
    # Convert geometry parameters to SI units
    # --------------------------------------------------------

    separation = gate_separation_nm * 1e-9
    depth = gate_depth_nm * 1e-9
    softening = softening_nm * 1e-9

    # --------------------------------------------------------
    # Gate positions
    # --------------------------------------------------------

    x_left = -separation
    x_plunger = 0.0
    x_right = separation

    # --------------------------------------------------------
    # Effective 3D distance from each gate
    # --------------------------------------------------------

    r_left = np.sqrt(
        (X - x_left) ** 2
        + Y ** 2
        + depth ** 2
        + softening ** 2
    )

    r_plunger = np.sqrt(
        (X - x_plunger) ** 2
        + Y ** 2
        + depth ** 2
        + softening ** 2
    )

    r_right = np.sqrt(
        (X - x_right) ** 2
        + Y ** 2
        + depth ** 2
        + softening ** 2
    )

    # --------------------------------------------------------
    # Effective gate influence
    # --------------------------------------------------------

    left_influence = 1.0 / r_left
    plunger_influence = 1.0 / r_plunger
    right_influence = 1.0 / r_right

    # --------------------------------------------------------
    # Normalize each influence profile
    #
    # This keeps the gate voltages as effective control
    # parameters rather than producing an arbitrary energy
    # scale directly from the 1/r kernel.
    # --------------------------------------------------------

    left_influence /= np.max(left_influence)
    plunger_influence /= np.max(plunger_influence)
    right_influence /= np.max(right_influence)

    # --------------------------------------------------------
    # Combine gate contributions
    # --------------------------------------------------------

    phi = (
        V_left * left_influence
        + V_plunger * plunger_influence
        + V_right * right_influence
    )

    return phi


# ============================================================
# ELECTRON POTENTIAL ENERGY
# ============================================================

def electron_potential_energy(
    V_left=-0.30,
    V_plunger=0.10,
    V_right=-0.30,
    gate_separation_nm=25.0,
    gate_depth_nm=20.0,
    softening_nm=2.0,
    transverse_strength=0.0
):
    """
    Convert the effective electrostatic potential into
    electron potential energy.

    U = -e * phi

    Returned energy is in meV.
    """

    e = 1.602176634e-19

    phi = gate_potential(
        V_left=V_left,
        V_plunger=V_plunger,
        V_right=V_right,
        gate_separation_nm=gate_separation_nm,
        gate_depth_nm=gate_depth_nm,
        softening_nm=softening_nm,
    )
    U = -e * phi

    # Joules -> meV
    U_meV = U / e * 1000.0
        # ============================================================
    # EFFECTIVE TRANSVERSE CONFINEMENT
    # ============================================================
    
    U_transverse = (
        transverse_strength
        * (Y / (Ly / 2)) ** 2
    )
    
    U_meV = U_meV + U_transverse

    # Choose the minimum of the potential as the zero of energy.
    # This removes an arbitrary constant energy offset without
    # changing the quantum-mechanical eigenstates.
    U_meV = U_meV - np.min(U_meV)

    return U_meV



# ============================================================
# POTENTIAL SUMMARY
# ============================================================

def summarize_potential(
    V_left=-0.30,
    V_plunger=0.10,
    V_right=-0.30,
    gate_separation_nm=25.0,
    gate_depth_nm=20.0,
    softening_nm=2.0,
):
    """
    Return basic numerical information about the
    calculated electrostatic potential.
    """

    phi = gate_potential(
        V_left=V_left,
        V_plunger=V_plunger,
        V_right=V_right,
        gate_separation_nm=gate_separation_nm,
        gate_depth_nm=gate_depth_nm,
        softening_nm=softening_nm,
    )

    U_meV = electron_potential_energy(
        V_left=V_left,
        V_plunger=V_plunger,
        V_right=V_right,
        transverse_strength=80.0,
        gate_separation_nm=gate_separation_nm,
        gate_depth_nm=gate_depth_nm,
        softening_nm=softening_nm,
    )

    return {
        "phi_min": np.min(phi),
        "phi_max": np.max(phi),
        "U_min_meV": np.min(U_meV),
        "U_max_meV": np.max(U_meV),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    summary = summarize_potential()

    print("Electrostatic model")
    print("===================")

    print(
        f"Potential range: "
        f"{summary['phi_min']:.6f} V → "
        f"{summary['phi_max']:.6f} V"
    )

    print(
        f"Electron energy range: "
        f"{summary['U_min_meV']:.6f} → "
        f"{summary['U_max_meV']:.6f} meV"
    )
