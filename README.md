# Quantum Dot Device Simulator

An interactive numerical simulator for a gate-defined semiconductor quantum dot, connecting device-level gate voltages to an effective confinement potential and the resulting quantum-mechanical energy states.

The project combines semiconductor device concepts, numerical quantum mechanics, and scientific computing in Python.

---

## Project Overview

Quantum dots are nanoscale semiconductor structures in which charge carriers are confined in space, producing discrete quantum energy levels.

This project develops a computational model of a **single-electron gate-defined quantum dot**. The simulator takes effective gate voltages as inputs, constructs a model confinement potential, and solves the two-dimensional time-independent Schrödinger equation to obtain the lowest-energy quantum states.

The goal is to build a bridge between:

**Gate voltages → Device potential → Quantum Hamiltonian → Energy spectrum → Wavefunctions**

The current implementation uses an effective reduced-order electrostatic model and a finite-difference Schrödinger solver.

---

## Current Model

The simulated device consists of three effective gates:

```text
Barrier 1       Plunger       Barrier 2
   │               │              │
   ▼               ▼              ▼

████████        ┌──────┐        ████████
                │  QD  │
████████        └──────┘        ████████
