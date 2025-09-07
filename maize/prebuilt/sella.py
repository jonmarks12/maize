from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node
import os

import sys
import torch
import numpy as np
from typing import Any
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
from sella import Sella
from torch import Tensor


class RunSellaTS(Node):
    """
    Run Sella PRFO with ASE calculator.

    Inputs:
        -TS_Guess: ase.Atoms
        -calculator: ase.calculator
        -run_directory: str
    """

    # inputs
    ts_guess = Input["ASEAtoms"]()
    run_directory = Input[str]()
    calculator = Input["ASECalculator"]()

    # outputs
    ts_out_loc = Output[str]()
    ts_out_atoms = Output["ASEAtoms"]()

    # parameters
    fmax: Parameter[float] = Parameter(default=5e-3)
    max_steps: Parameter[int] = Parameter(default=50)

    def run(self) -> None:
        """Run Sella TS refinement"""
        ts_guess = self.ts_guess.receive()
        run_directory = self.run_directory.receive()
        calculator = self.calculator.receive()

        charge = np.sum(ts_guess.get_initial_charges())
        multiplicity = np.sum(ts_guess.get_initial_magnetic_moments()) + 1

        ts_guess.calc = calculator
        dyn = Sella(ts_guess, trajectory=os.path.join(run_directory, "sella.traj"))
        dyn.run(self.fmax.value, self.max_steps.value)
        write(os.path.join(run_directory, "sella_guess.xyz"), ts_guess, format="xyz")

        self.ts_out_loc.send(run_directory)
        self.ts_out_atoms.send(ts_guess)
