import numpy as np

from electrostatics import (
    X,
    Y,
    gate_potential,
    electron_potential_energy,
)


# ============================================================
# TEST 1 — LEFT-RIGHT SYMMETRY
# ============================================================

def test_symmetry():

    phi = gate_potential(
        V_left=-0.30,
        V_plunger=-0.20,
        V_right=-0.30,
    )

    # Mirror the potential about x = 0
    phi_mirror = np.fliplr(phi)

    symmetry_error = np.max(
        np.abs(phi - phi_mirror)
    )

    print("\nTEST 1 — Symmetry")
    print("-----------------")
    print(f"Maximum symmetry error: {symmetry_error:.6e}")

    assert symmetry_error < 1e-12

    print("PASS")


# ============================================================
# TEST 2 — ASYMMETRIC GATES
# ============================================================

def test_asymmetry():

    phi = gate_potential(
        V_left=-0.40,
        V_plunger=-0.20,
        V_right=-0.20,
    )

    phi_mirror = np.fliplr(phi)

    asymmetry_error = np.max(
        np.abs(phi - phi_mirror)
    )

    print("\nTEST 2 — Asymmetric gate response")
    print("----------------------------------")
    print(f"Asymmetric-device error: {asymmetry_error:.6e}")

    assert asymmetry_error > 1e-3

    print("PASS")


# ============================================================
# TEST 3 — PLUNGER RESPONSE
# ============================================================

def test_plunger_response():

    # Look at the center of the device
    center_index_x = X.shape[1] // 2
    center_index_y = X.shape[0] // 2

    phi_values = []

    plunger_values = [
        -0.10,
        -0.20,
        -0.30,
    ]

    for V_plunger in plunger_values:

        phi = gate_potential(
            V_left=-0.30,
            V_plunger=V_plunger,
            V_right=-0.30,
        )

        center_phi = phi[
            center_index_y,
            center_index_x
        ]

        phi_values.append(center_phi)

    print("\nTEST 3 — Plunger response")
    print("--------------------------")

    for V, phi in zip(plunger_values, phi_values):
        print(
            f"V_plunger = {V:.2f} V   "
            f"center potential = {phi:.6f} V"
        )

    # More negative plunger voltage should produce
    # a more negative effective potential at the center.
    assert phi_values[0] > phi_values[1]
    assert phi_values[1] > phi_values[2]

    print("PASS")


# ============================================================
# TEST 4 — ELECTROSTATIC POTENTIAL → ELECTRON ENERGY
# ============================================================

# ============================================================
# TEST 4 — ELECTROSTATIC POTENTIAL → ELECTRON ENERGY
# ============================================================

def test_energy_conversion():

    phi = gate_potential(
        V_left=-0.30,
        V_plunger=-0.20,
        V_right=-0.30,
    )

    U_meV = electron_potential_energy(
        V_left=-0.30,
        V_plunger=-0.20,
        V_right=-0.30,
    )

    # For an electron:
    #
    # U = -e * phi
    #
    # Converting to meV:
    #
    # U_meV = -phi * 1000
    #
    # The potential energy is then shifted so that its
    # minimum is defined as zero.

    expected_U_meV = -phi * 1000.0

    expected_U_meV = (
        expected_U_meV
        - np.min(expected_U_meV)
    )

    conversion_error = np.max(
        np.abs(U_meV - expected_U_meV)
    )

    print("\nTEST 4 — Energy conversion")
    print("---------------------------")
    print(
        f"Maximum conversion error: "
        f"{conversion_error:.6e} meV"
    )

    assert conversion_error < 1e-12

    print("PASS")


# ============================================================
# TEST 5 — FINITE NUMERICAL VALUES
# ============================================================

def test_finite_values():

    phi = gate_potential(
        V_left=-0.30,
        V_plunger=-0.20,
        V_right=-0.30,
    )

    U_meV = electron_potential_energy(
        V_left=-0.30,
        V_plunger=-0.20,
        V_right=-0.30,
    )

    print("\nTEST 5 — Numerical validity")
    print("----------------------------")

    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(U_meV))

    print("No NaN or infinite values found")
    print("PASS")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==========================================")
    print("   ELECTROSTATICS MODEL VALIDATION")
    print("==========================================")

    test_symmetry()
    test_asymmetry()
    test_plunger_response()
    test_energy_conversion()
    test_finite_values()

    print("\n==========================================")
    print("ALL ELECTROSTATICS TESTS PASSED")
    print("==========================================")
