# Quantum Dot Device Simulator

An interactive numerical simulator for a gate-defined semiconductor quantum dot, connecting effective gate controls to a confinement potential and the resulting quantum-mechanical states.

The project combines semiconductor device concepts, numerical quantum mechanics, charge-state modeling, and scientific computing in Python.

---

## Project Overview

Quantum dots are nanoscale semiconductor structures in which charge carriers are confined in space, producing discrete quantum energy levels.

This project develops a computational framework for studying **gate-defined semiconductor quantum dots**. The current implementation contains models for:

- Single quantum-dot confinement
- Double quantum dots
- Quantum-state energy spectra
- Ground-state wavefunctions and probability densities
- Gate-controlled detuning
- Tunnel-splitting behavior
- Charge stability diagrams
- Triple-point identification and validation

The long-term goal is to build a computational bridge between:

**Gate geometry / controls → Electrostatic potential → Quantum Hamiltonian → Energy spectrum → Wavefunctions → Charge states**

The present implementation is being developed incrementally, starting from reduced-order effective models and progressing toward more physically explicit electrostatic modeling.

---

## Current Model

The initial single-dot simulator uses three effective gate controls:

```text
Barrier 1       Plunger       Barrier 2
   │               │              │
   ▼               ▼              ▼

████████        ┌──────┐        ████████
                │  QD  │
████████        └──────┘        ████████
