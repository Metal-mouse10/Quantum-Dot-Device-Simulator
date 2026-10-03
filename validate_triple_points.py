from triple_points import find_triple_point
from charge_stability import charge_energy


def validate_triple_point(N_left, N_right):

    (
        V_left,
        V_right,
        n_g_left,
        n_g_right,
    ) = find_triple_point(
        N_left,
        N_right,
    )

    # Three states expected to meet
    states = [
        (N_left, N_right),
        (N_left + 1, N_right),
        (N_left, N_right + 1),
    ]

    energies = {}

    for state in states:

        energies[state] = charge_energy(
            state[0],
            state[1],
            V_left,
            V_right,
        )

    lowest_energy = min(energies.values())

    # Check degeneracy
    energy_spread = (
        max(energies.values())
        - min(energies.values())
    )

    # Check whether another nearby state
    # is lower than the three candidate states.
    nearby_states = []

    for left in range(
        max(0, N_left - 1),
        N_left + 3,
    ):

        for right in range(
            max(0, N_right - 1),
            N_right + 3,
        ):

            nearby_states.append(
                (left, right)
            )

    global_minimum = min(
        charge_energy(
            left,
            right,
            V_left,
            V_right,
        )
        for left, right in nearby_states
    )

    is_ground_state_triple_point = (
        energy_spread < 1e-10
        and abs(
            lowest_energy
            - global_minimum
        ) < 1e-10
    )

    return (
        V_left,
        V_right,
        energies,
        energy_spread,
        is_ground_state_triple_point,
    )


if __name__ == "__main__":

    candidate_points = [
        (0, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    print()
    print("Triple-Point Validation")
    print("=======================")

    for N_left, N_right in candidate_points:

        (
            V_left,
            V_right,
            energies,
            spread,
            valid,
        ) = validate_triple_point(
            N_left,
            N_right,
        )

        print()
        print(
            f"Reference state: "
            f"({N_left},{N_right})"
        )

        print(
            f"Gate point: "
            f"({V_left:.6f}, "
            f"{V_right:.6f}) V"
        )

        for state, energy in energies.items():

            print(
                f"E{state} = "
                f"{energy:.10f} meV"
            )

        print(
            f"Energy spread = "
            f"{spread:.3e} meV"
        )

        print(
            f"Ground-state triple point: "
            f"{valid}"
        )
