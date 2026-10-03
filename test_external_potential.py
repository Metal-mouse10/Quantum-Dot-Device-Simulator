import numpy as np

from electrostatics import (
    X,
    Y,
    electron_potential_energy,
)

from quantum_dot import (
    solve_external_potential,
)


# ============================================================
# TEST — ELECTROSTATICS → SCHRÖDINGER COUPLING
# ============================================================

def test_external_potential_solver():

    print("\n")
    print("==========================================")
    print(" EXTERNAL POTENTIAL SCHRÖDINGER TEST")
    print("==========================================")


    # --------------------------------------------------------
    # Generate electrostatic potential
    # --------------------------------------------------------

    U_meV = electron_potential_energy(
        V_left=-0.30,
        V_plunger=0.10,
        V_right=-0.30,
    )

    print("\nExternal potential")
    print("------------------")
    print(f"Shape: {U_meV.shape}")
    print(
        f"Energy range: "
        f"{np.min(U_meV):.6f} → "
        f"{np.max(U_meV):.6f} meV"
    )


    # --------------------------------------------------------
    # Solve Schrödinger equation
    # --------------------------------------------------------

    energies, states = solve_external_potential(
        U_meV=U_meV,
        x_grid=X[0, :],
        y_grid=Y[:, 0],
        num_states=4,
    )


    # --------------------------------------------------------
    # Convert energies to meV
    # --------------------------------------------------------

    eV = 1.602176634e-19

    energies_meV = (
        energies / eV * 1000.0
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nQuantum states")
    print("--------------")

    for i, energy in enumerate(energies_meV):

        print(
            f"E{i} = {energy:.6f} meV"
        )


    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    expected_grid_size = (
        len(X[0, :]) * len(Y[:, 0])
    )

    assert states.shape == (
        expected_grid_size,
        4
    )


    # --------------------------------------------------------
    # Validate finite values
    # --------------------------------------------------------

    assert np.all(
        np.isfinite(energies)
    )

    assert np.all(
        np.isfinite(states)
    )


    # --------------------------------------------------------
    # Validate energy ordering
    # --------------------------------------------------------

    assert np.all(
        np.diff(energies) >= 0
    )


    print("\nValidation")
    print("----------")
    print("State-array dimensions       PASS")
    print("Finite energies              PASS")
    print("Finite wavefunctions         PASS")
    print("Energy ordering              PASS")

    print("\n==========================================")
    print("EXTERNAL POTENTIAL TEST PASSED")
    print("==========================================")


if __name__ == "__main__":
    test_external_potential_solver()
