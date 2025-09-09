from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node
from typing import Dict, Any
import os
import subprocess

import numpy as np
import ase


class RunPRFO(Node):
    """
    Run QChem PRFO TS search.

    Inputs:
        -TS_Guess: ase.Atoms
    Outputs:
        -None
    """

    ts_guess = Input["ASEAtoms"]()
    run_directory = Input[str]()
    output_success = Output[Dict[str, Any]]()
    output_failed = Output[Dict[str, Any]]()

    method: Parameter[str] = Parameter(default="wb97x-v")
    basis: Parameter[str] = Parameter(default="def2-tzvp")
    num_threads: Parameter[int] = Parameter(default=16)

    @staticmethod
    def _processtsout(outfile) -> Dict:
        opt_cycles = []
        opt_conv = False
        success = False
        im_freqs = []
        energy = None
        opt_e = 0
        with open(outfile, "r") as f:
            data = f.readlines()
            for i, line in enumerate(data):
                if "Final energy is" in line:
                    energy = float(line.split()[-1])
                    success = True
                    opt_conv = True
                if "nergy is  " in line:
                    opt_e = float(line.split()[-1])
                if "   Optimization Cycle:" in line:
                    opt_cycles.append(int(line.split()[-1]))
                if "INFRARED INTENSITIES (KM/MOL)" in line:
                    freqs = data[i + 6].split()[1:]
        if float(freqs[0]) >= 0.0:
            success = False
        if float(freqs[1]) <= 0.0:
            success = False
        return {
            "Success": success,
            "Opt Convergence": opt_conv,
            "Final Energy": energy,
            "Opt Cycles": opt_cycles,
            "Final Frequencies": freqs,
        }

    def _writetsqcin(self, structure, filename, chg, mult) -> None:
        """
        Writes and submits a TS.qcin file to run the PRFO

        TODO: Make this significantly more flexible through parameters and inputs where appropriate
        """
        chem_symb = structure.get_chemical_symbols()
        coordinates = structure.get_positions()
        with open(filename, "w") as f:
            f.write(f"$molecule\n{chg} {mult}\n")
            for i in range(len(chem_symb)):
                f.write(chem_symb[i])
                f.write(" ")
                for coord in coordinates[i]:
                    f.write(str(coord))
                    f.write(" ")
                f.write("\n")
            f.write(
                "$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n"
                "basis {}\n"
                "scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n"
                "WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n@@@\n\n".format(
                    self.method.value, self.basis.value
                )
            )
            f.write(
                "$rem\nJOBTYPE       TS\nMETHOD       {}\n"
                "BASIS       {}\n"
                "scf_max_cycles 250\ngeom_opt_max_cycles 250\ngeom_opt_hessian read\nscf_guess read\n"
                "mem_total 40000\nmem_static 6000\n"
                "WAVEFUNCTION_ANALYSIS       FALSE\n$end\n\n$molecule\nread\n$end\n\n@@@\n\n".format(
                    self.method.value, self.basis.value
                )
            )
            f.write(
                "$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n"
                "basis {}\n"
                "scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n"
                "WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n$molecule\nread\n$end\n".format(
                    self.method.value, self.basis.value
                )
            )

    def run(self) -> None:
        ts_guess = self.ts_guess.receive()
        run_directory = self.run_directory.receive()

        # some calculators store charges this way
        charge = int(np.sum(ts_guess.get_initial_charges()))
        multiplicity = int(np.sum(ts_guess.get_initial_magnetic_moments()) + 1)
        filename = os.path.join(run_directory,"ts_guess.qcin")
        print(f"Filename: {filename}")
        self._writetsqcin(structure=ts_guess, filename=filename, chg=charge, mult=multiplicity)
        subprocess.run(
            ["qchem", "-nt", str(self.num_threads.value), filename, f"{filename}.out"], check=True
        )
        prfo_res = self._processtsout(f"{filename}.out")
        if prfo_res["Success"]:
            self.output_success.send(prfo_res)
        if not prfo_res["Success"]:
            self.output_failed.send(prfo_res)
