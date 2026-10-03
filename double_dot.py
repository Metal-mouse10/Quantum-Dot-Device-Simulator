import numpy as np

from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

hbar = 1.054571817e-34
eV = 1.602176634e-19
m_e = 9.1093837e-31

# GaAs effective electron mass
m_eff = 0.067 * m_e


# ============================================================
# DEVICE GRID
# ============================================================

Lx = 100e-9
Ly = 100e-9

# Keep this at 150 initially for interactive calculations
Nx = 150
Ny = 150

x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)

X, Y = np.meshgrid(x, y)


# ============================================================
# DOUBLE QUANTUM DOT POTENTIAL
# ============================================================

def build_double_dot_potential(
    V_left=-0.20,
    V_right=-0.20,
    V_barrier=-0.30,
    dot_separation_nm=30.0,
    dot_width_nm=12.0,
    barrier_width_nm=6.0,
):
    """
    Construct an effective double-quantum-dot potential.

    Parameters
    ----------
    V_left : float
        Effective left-dot control parameter in volts.

    V_right : float
        Effective right-dot control parameter in volts.

    V_barrier : float
        Effective barrier-gate parameter in volts.

    dot_separation_nm : float
        Distance between the centers of the two dots.

    dot_width_nm : float
        Width of each attractive dot potential.

    barrier_width_nm : float
        Width of the central barrier.

    Returns
    -------
    U_meV : ndarray
        Effective potential in meV.

    U : ndarray
        Effective potential in joules.

    Notes
    -----
    This is a phenomenological effective potential.

    The gate parameters are NOT experimentally calibrated
    voltages and do not represent a full electrostatic
    solution of a fabricated device.
    """

    # Convert nm → m
    separation = dot_separation_nm * 1e-9
    sigma_dot = dot_width_nm * 1e-9
    sigma_barrier = barrier_width_nm * 1e-9

    # Dot positions
    x_left = -separation / 2
    x_right = separation / 2

    # --------------------------------------------------------
    # Dot confinement strengths
    # --------------------------------------------------------

    # Phenomenological energy scale
    left_depth = (-V_left) * 50.0
    right_depth = (-V_right) * 50.0

    # --------------------------------------------------------
    # Left quantum dot
    # --------------------------------------------------------

    U_left = (
        -left_depth
        * np.exp(
            -(
                (X - x_left) ** 2
                + Y ** 2
            )
            / (2 * sigma_dot ** 2)
        )
    )

    # --------------------------------------------------------
    # Right quantum dot
    # --------------------------------------------------------

    U_right = (
        -right_depth
        * np.exp(
            -(
                (X - x_right) ** 2
                + Y ** 2
            )
            / (2 * sigma_dot ** 2)
        )
    )

    # --------------------------------------------------------
    # Central tunnel barrier
    # --------------------------------------------------------

    barrier_height = (-V_barrier) * 100.0

    U_barrier = (
        barrier_height
        * np.exp(
            -(X ** 2)
            / (2 * sigma_barrier ** 2)
        )
        * np.exp(
            -(Y ** 2)
            / (2 * (12e-9) ** 2)
        )
    )

    # --------------------------------------------------------
    # Weak transverse confinement
    # --------------------------------------------------------

    U_transverse = (
        1.5
        * (Y / (10e-9)) ** 2
    )

    # --------------------------------------------------------
    # Total effective potential
    # --------------------------------------------------------

    U_meV = (
        U_left
        + U_right
        + U_barrier
        + U_transverse
    )

    # Convert meV → joules
    U = U_meV * 1e-3 * eV

    return U_meV, U


# ============================================================
# DOUBLE-DOT SCHRÖDINGER SOLVER
# ============================================================

def solve_double_dot(
    V_left=-0.20,
    V_right=-0.20,
    V_barrier=-0.30,
    dot_separation_nm=30.0,
    dot_width_nm=12.0,
    barrier_width_nm=6.0,
    num_states=6,
):
    """
    Solve the 2D effective-mass Schrödinger equation
    for a double quantum dot.

    Returns
    -------
    U_meV : ndarray
        Effective potential in meV.

    energies : ndarray
        Eigenenergies in joules.

    states : ndarray
        Corresponding eigenstates.
    """

    # Grid spacing
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    N_x = len(x)
    N_y = len(y)

    # --------------------------------------------------------
    # Finite-difference second derivatives
    # --------------------------------------------------------

    main_x = -2.0 * np.ones(N_x)
    off_x = np.ones(N_x - 1)

    D2_x = diags(
        [off_x, main_x, off_x],
        [-1, 0, 1],
        shape=(N_x, N_x),
    ) / dx**2

    main_y = -2.0 * np.ones(N_y)
    off_y = np.ones(N_y - 1)

    D2_y = diags(
        [off_y, main_y, off_y],
        [-1, 0, 1],
        shape=(N_y, N_y),
    ) / dy**2

    I_x = eye(N_x)
    I_y = eye(N_y)

    # --------------------------------------------------------
    # 2D Laplacian
    # --------------------------------------------------------

    laplacian = (
        kron(I_y, D2_x)
        + kron(D2_y, I_x)
    )

    # --------------------------------------------------------
    # Kinetic energy
    # --------------------------------------------------------

    T = (
        -(hbar**2)
        / (2 * m_eff)
        * laplacian
    )

    # --------------------------------------------------------
    # Potential
    # --------------------------------------------------------

    U_meV, U = build_double_dot_potential(
        V_left=V_left,
        V_right=V_right,
        V_barrier=V_barrier,
        dot_separation_nm=dot_separation_nm,
        dot_width_nm=dot_width_nm,
        barrier_width_nm=barrier_width_nm,
    )

    # --------------------------------------------------------
    # Hamiltonian
    # --------------------------------------------------------

    H = T + diags(U.ravel())

    # --------------------------------------------------------
    # Lowest-energy states
    # --------------------------------------------------------

    energies, states = eigsh(
        H,
        k=num_states,
        which="SA",
    )

    # Sort eigenvalues
    idx = np.argsort(energies)

    energies = energies[idx]
    states = states[:, idx]

    return U_meV, energies, states


# ============================================================
# UTILITY: ENERGY CONVERSION
# ============================================================

def energies_to_meV(energies):
    """
    Convert energies from joules to meV.
    """

    return energies / eV * 1e3


# ============================================================
# UTILITY: WAVEFUNCTION PROBABILITY DENSITY
# ============================================================

def probability_density(state):
    """
    Convert a discrete eigenvector into a continuum-normalized
    probability density in m^-2.
    """

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    psi = state.reshape(X.shape)

    probability = (
        np.abs(psi) ** 2
        / (dx * dy)
    )

    return probability


# ============================================================
# UTILITY: TUNNEL SPLITTING
# ============================================================

def tunnel_splitting(energies):
    """
    Calculate the splitting between the two lowest states.

    Delta E = E1 - E0

    Returns
    -------
    float
        Tunnel splitting in meV.
    """

    energies_meV = energies_to_meV(energies)

    return energies_meV[1] - energies_meV[0]


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":

    U_meV, energies, states = solve_double_dot(
        V_left=-0.20,
        V_right=-0.20,
        V_barrier=-0.30,
        dot_separation_nm=30,
        num_states=6,
    )

    energies_meV = energies_to_meV(energies)

    print("\nDouble Quantum Dot")
    print("------------------")

    for i, energy in enumerate(energies_meV):
        print(
            f"E{i} = {energy:.6f} meV"
        )

    print(
        f"\nTunnel splitting = "
        f"{tunnel_splitting(energies):.6f} meV"
    )
