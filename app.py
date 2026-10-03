import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from quantum_dot import (
    solve_effective_device,
    X,
    Y,
    eV
)

from electrostatics import (
    X as X_electrostatic,
    Y as Y_electrostatic,
    electron_potential_energy
)
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Dot Device Simulator",
    page_icon="⚛️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚛️ Quantum Dot Device Simulator")

st.markdown(
    """
    ### Gate-defined single-electron quantum dot

    An interactive numerical model connecting gate voltages,
    effective confinement potential, and quantum-mechanical
    energy states through a finite-difference Schrödinger solver.
    """
)

st.divider()


# ============================================================
# SIDEBAR — DEVICE PARAMETERS
# ============================================================

st.sidebar.header("Device Parameters")

V_gate1 = st.sidebar.slider(
    "Barrier 1 voltage (V)",
    min_value=-0.50,
    max_value=-0.05,
    value=-0.30,
    step=0.01
)

V_plunger = st.sidebar.slider(
    "Plunger voltage (V)",
    min_value=-0.50,
    max_value=-0.05,
    value=-0.20,
    step=0.01
)

V_gate2 = st.sidebar.slider(
    "Barrier 2 voltage (V)",
    min_value=-0.50,
    max_value=-0.05,
    value=-0.30,
    step=0.01
)

num_states = st.sidebar.slider(
    "Number of quantum states",
    min_value=2,
    max_value=6,
    value=4,
    step=1
)


st.sidebar.divider()

st.sidebar.markdown(
    """
    **Device**

    100 × 100 nm domain

    GaAs effective mass

    Single-electron model

    Finite-difference Schrödinger solver
    """
)


# ============================================================
# SOLVE QUANTUM SYSTEM
# ============================================================

with st.spinner("Solving quantum dot..."):

    U_meV, energies, states = solve_effective_device(
        V_gate1=V_gate1,
        V_plunger=V_plunger,
        V_gate2=V_gate2,
        num_states=num_states
    )
    # ============================================================
    # ELECTROSTATIC POTENTIAL
    # ============================================================
    
    U_electrostatic_meV = electron_potential_energy(
        V_left=V_gate1,
        V_plunger=V_plunger,
        V_right=V_gate2
    )


# Convert energy from joules to meV

energy_meV = energies / eV * 1e3


# ============================================================
# ENERGY SPECTRUM
# ============================================================

st.subheader("Quantum Energy Spectrum")

cols = st.columns(num_states)

for i, col in enumerate(cols):

    col.metric(
        f"E{i}",
        f"{energy_meV[i]:.3f} meV"
    )


# ============================================================
# POTENTIAL + GROUND STATE
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# EFFECTIVE DEVICE POTENTIAL
# ------------------------------------------------------------

with col1:

    st.subheader("Effective Device Potential")

    fig, ax = plt.subplots(figsize=(7, 5))

    mesh = ax.pcolormesh(
        X * 1e9,
        Y * 1e9,
        U_meV,
        shading="auto"
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("Gate-defined potential energy")

    fig.colorbar(
        mesh,
        ax=ax,
        label="Energy (meV)"
    )

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ------------------------------------------------------------
# GROUND STATE
# ------------------------------------------------------------

with col2:

    st.subheader("Ground-State Probability Density")

    psi0 = states[:, 0].reshape(X.shape)
    dx_nm = (X[0, 1] - X[0, 0]) * 1e9
    probability = np.abs(psi0) ** 2 / dx_nm**2

    fig, ax = plt.subplots(figsize=(7, 5))

    mesh = ax.pcolormesh(
        X * 1e9,
        Y * 1e9,
        probability,
        shading="auto"
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(r"$|\psi_0(x,y)|^2$")

    fig.colorbar(
        mesh,
        ax=ax,
        label=r"Probability density (nm$^{-2}$)"
    )

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

# ============================================================
# ELECTROSTATIC POTENTIAL
# ============================================================

st.divider()

st.subheader("Electrostatic Potential")

st.markdown(
    """
    The applied gate voltages are mapped onto an effective
    electrostatic potential at the semiconductor plane.
    """
)

fig, ax = plt.subplots(figsize=(8, 5))

mesh = ax.pcolormesh(
    X_electrostatic * 1e9,
    Y_electrostatic * 1e9,
    U_electrostatic_meV,
    shading="auto"
)

ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title("Effective Electrostatic Electron Potential")

fig.colorbar(
    mesh,
    ax=ax,
    label="Energy (meV)"
)

st.pyplot(fig, use_container_width=True)

plt.close(fig)
col1, col2, col3 = st.columns(3)

col1.metric(
    "Minimum potential",
    f"{np.min(U_electrostatic_meV):.2f} meV"
)

col2.metric(
    "Maximum potential",
    f"{np.max(U_electrostatic_meV):.2f} meV"
)

col3.metric(
    "Grid",
    f"{U_electrostatic_meV.shape[0]} × {U_electrostatic_meV.shape[1]}"
)


# ============================================================
# ENERGY LEVEL DIAGRAM
# ============================================================

st.subheader("Energy Levels")

fig, ax = plt.subplots(figsize=(8, 4))

for i, E in enumerate(energy_meV):

    ax.hlines(
        E,
        0,
        1,
        linewidth=3
    )

    ax.text(
        1.03,
        E,
        f"E{i} = {E:.3f} meV",
        va="center"
    )


ax.set_xlim(0, 1.5)
ax.set_xticks([])

ax.set_ylabel("Energy (meV)")

ax.set_title("Lowest-Energy Quantum States")

st.pyplot(fig, use_container_width=True)

plt.close(fig)


# ============================================================
# DEVICE CONCEPT
# ============================================================

st.divider()

st.subheader("Device Concept")

st.markdown(
    """
    **Barrier 1 → Plunger → Barrier 2**

    The gate voltages are mapped onto an effective confinement
    potential. The resulting potential is incorporated into a
    finite-difference single-particle Hamiltonian, from which
    the lowest quantum states are calculated.
    """
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model assumptions, limitations and future extensions"):

    st.markdown(
        """
        ### Current model

        This simulator models a single electron in a two-dimensional
        effective gate-defined confinement potential.

        The quantum states are obtained by solving the effective-mass
        Schrödinger equation using a finite-difference discretization
        and a sparse eigenvalue solver.

        ### Electrostatic model

        The relationship between the applied gate parameters and the
        effective potential is phenomenological. The present model does
        not solve the full three-dimensional electrostatic Poisson
        equation and is not calibrated to a specific fabricated device.

        Therefore, the gate parameters should be interpreted as
        effective model parameters rather than experimentally calibrated
        device voltages.

        ### Physical parameters

        - Device domain: 100 × 100 nm
        - Semiconductor: GaAs
        - Electron effective mass: 0.067 mₑ
        - Single-electron approximation
        - Two-dimensional finite-difference grid

        ### Future extensions

        - Realistic 3D gate geometry
        - Electrostatic Poisson solver
        - Self-consistent Schrödinger-Poisson calculation
        - Double quantum dots
        - Tunnel coupling
        - Charge stability diagrams
        - Material and device-specific parameters
        - Comparison with experimentally calibrated devices
        """
    )
