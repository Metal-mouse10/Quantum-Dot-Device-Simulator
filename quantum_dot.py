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

# Device dimensions
Lx = 100e-9
Ly = 100e-9

# Grid resolution
# Reduced from the research notebook for faster interactive use
Nx = 150
Ny = 150

x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)

X, Y = np.meshgrid(x, y)


# ============================================================
# EFFECTIVE GATE-DEFINED POTENTIAL
# ============================================================

def build_effective_device_potential(
    V_gate1=-0.30,
    V_plunger=-0.20,
    V_gate2=-0.30,
    sigma_barrier_nm=6,
    sigma_plunger_nm=18
):
    """
    Construct an effective gate-defined quantum-dot potential.

    Parameters
    ----------
    V_gate1 : float
        Left barrier gate voltage in volts.

    V_plunger : float
        Central plunger gate voltage in volts.

    V_gate2 : float
        Right barrier gate voltage in volts.

    Returns
    -------
    U_meV : ndarray
        Effective potential in meV.

    U : ndarray
        Effective potential in joules.
    """

    sigma_b = sigma_barrier_nm * 1e-9
    sigma_p = sigma_plunger_nm * 1e-9

    # Phenomenological gate-to-energy coupling
    barrier_energy_1 = (-V_gate1) * 100.0
    barrier_energy_2 = (-V_gate2) * 100.0

    plunger_energy = (-V_plunger) * 50.0

    # --------------------------------------------------------
    # Left barrier
    # --------------------------------------------------------

    U_left = (
        barrier_energy_1
        * np.exp(
            -((X + 25e-9) ** 2)
            / (2 * sigma_b ** 2)
        )
        * np.exp(
            -(Y ** 2)
            / (2 * (12e-9) ** 2)
        )
    )

    # --------------------------------------------------------
    # Right barrier
    # --------------------------------------------------------

    U_right = (
        barrier_energy_2
        * np.exp(
            -((X - 25e-9) ** 2)
            / (2 * sigma_b ** 2)
        )
        * np.exp(
            -(Y ** 2)
            / (2 * (12e-9) ** 2)
        )
    )

    # --------------------------------------------------------
    # Central attractive plunger
    # --------------------------------------------------------

    U_plunger = (
        -plunger_energy
        * np.exp(
            -(X ** 2 + Y ** 2)
            / (2 * sigma_p ** 2)
        )
    )

    # --------------------------------------------------------
    # Weak transverse confinement
    # --------------------------------------------------------

    U_transverse = (
        1.5
        * (Y / (10e-9)) ** 2
    )

    # Total effective potential
    U_meV = (
        U_left
        + U_right
        + U_plunger
        + U_transverse
    )

    # Convert meV → joules
    U = U_meV * 1e-3 * eV

    return U_meV, U


# ============================================================
# SCHRÖDINGER SOLVER
# ============================================================

def solve_effective_device(
    V_gate1=-0.30,
    V_plunger=-0.20,
    V_gate2=-0.30,
    num_states=4
):
    """
    Solve the 2D single-electron Schrödinger equation.

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
    N_grid = X.shape[0]
    dx = X[0, 1] - X[0, 0]

    # --------------------------------------------------------
    # 1D finite-difference second derivative
    # --------------------------------------------------------

    main = -2.0 * np.ones(N_grid)
    off = np.ones(N_grid - 1)

    D2 = diags(
        [off, main, off],
        [-1, 0, 1],
        shape=(N_grid, N_grid)
    ) / dx**2

    I = eye(N_grid)

    # --------------------------------------------------------
    # 2D Laplacian
    # --------------------------------------------------------

    laplacian = (
        kron(I, D2)
        + kron(D2, I)
    )

    # --------------------------------------------------------
    # Kinetic energy operator
    # --------------------------------------------------------

    T = (
        -(hbar**2)
        / (2 * m_eff)
        * laplacian
    )

    # --------------------------------------------------------
    # Effective device potential
    # --------------------------------------------------------

    U_meV, U = build_effective_device_potential(
        V_gate1=V_gate1,
        V_plunger=V_plunger,
        V_gate2=V_gate2
    )

    # --------------------------------------------------------
    # Hamiltonian
    # --------------------------------------------------------

    H = T + diags(U.ravel())

    # --------------------------------------------------------
    # Lowest-energy eigenstates
    # --------------------------------------------------------

    energies, states = eigsh(
        H,
        k=num_states,
        which="SA"
    )

    # Sort eigenvalues
    idx = np.argsort(energies)

    energies = energies[idx]
    states = states[:, idx]

    return U_meV, energies, states
