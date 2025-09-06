from typing import Optional
import os
from ase.io import read
from maize.core.interface import Parameter, Output, Input, MultiOutput
from maize.core.node import Node

class _FeedCalculator(Node):
    calculator = Parameter[str]()
    out = Output["ASECalculator"]()
    out_sella = Output["ASECalculator"]()
    def run(self) -> None:
        # Load calculator
        if self.calculator.value == "qchem":
            from ase.calculators.qchem import QChem

            calc = QChem(
                label="fsm",
                method="wb97x-v",
                basis="def2-tzvp",
                charge=chg,
                multiplicity=mult,
                sym_ignore="true",
                symmetry="false",
                scf_algorithm="diis_gdm",
                scf_max_cycles="500",
                nt=nt,
            )
        elif self.calculator.value == "xtb":
            from xtb.ase.calculator import XTB  # type: ignore [import-not-found]

            calc = XTB(method="GFN2-xTB")
        elif self.calculator.value == "uma_s":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=dev)
            calc = FAIRChemCalculator(predictor, task_name="omol")
        elif self.calculator.value == "uma_m":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
            calc = FAIRChemCalculator(predictor, task_name="omol")
        elif self.calculator.value == "eSEN":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device=dev)
            calc = FAIRChemCalculator(predictor)
        elif self.calculator.value == "aimnet2":
            from aimnet2calc import AIMNet2ASE  # type: ignore [import-not-found]

            calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
        elif self.calculator.value == "emt":
            from ase.calculators.emt import EMT

            calc = EMT()
        elif self.calculator.value == "maceomol":
            import torch
            from mace.calculators import mace_omol

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            calc = mace_omol(model="extra_large",device=dev)
        else:
            raise ValueError(f"Unknown calculator {calculator}")
        calc2 = calc
        self.out_sella.send(calc2)
        self.out.send(calc)

class _FeedAtoms(Node):
    path: Parameter[str] = Parameter()
    out: Output["ASEAtoms"] = Output()
    def run(self) -> None:
        from ase.io import read
        atoms = read(self.path.value)
        self.out.send(atoms)

class _FeedInitial(Node):
    path: Parameter[str] = Parameter()
    reactant: Output["ASEAtoms"] = Output()
    product: Output["ASEAtoms"] = Output()

    def run(self) -> None:
        from ase.io import read
        atoms = read(self.path.value,index=':')
        reaction_dir = os.path.dirname(self.path.value)
        with open(os.path.join(reaction_dir,"chg")) as f:
            chg = int(f.read())
        with open(os.path.join(reaction_dir,"mult")) as f:
             mult = int(f.read())
        atoms[0].info.update({'charge': chg, 'spin': mult}) #works for FAIRchem models
        atoms[1].info.update({'charge': chg, 'spin': mult})

        #update the actual ase information
        chg_list = atoms[0].get_initial_charges()
        chg_list[0] = chg
        mult_list = atoms[0].get_initial_magnetic_moments()
        mult_list[0] = mult-1

        atoms[0].set_initial_magnetic_moments(mult_list)
        atoms[0].set_initial_charges(chg_list)

        atoms[1].set_initial_magnetic_moments(mult_list)
        atoms[1].set_initial_charges(chg_list)
        self.reactant.send(atoms[0])
        self.product.send(atoms[1])