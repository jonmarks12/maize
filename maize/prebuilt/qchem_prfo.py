from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node
import os

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
    
    method: Parameter[str] = Parameter(default = "wb97x-v")
    basis: Parameter[str] = Parameter(default = "def2-tzvp")
    num_threads: Parameter[int] = Parameter(default = 16)

    def _writetsqcin(self,structure,filename,chg,mult) -> None:
        """
        Writes and submits a TS.qcin file to run the PRFO

        TODO: Make this significantly more flexible through parameters and inputs where appropriate
        """
        chem_symb = structure.get_chemical_symbols()
        coordinates = structure.get_positions()
        with open(filename,'w') as f:
            f.write(f'$molecule\n{chg} {mult}\n')
            for i in range(len(chem_symb)):
                f.write(chem_symb[i])
                f.write(' ')
                for coord in coordinates[i]:
                    f.write(str(coord))
                    f.write(' ')
                f.write('\n')
            f.write('$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n'
                    'basis {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n@@@\n\n'.format(self.method.value,self.basis.value))
            f.write('$rem\nJOBTYPE       TS\nMETHOD       {}\n'
                    'BASIS       {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\ngeom_opt_hessian read\nscf_guess read\n'
                    'mem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS       FALSE\n$end\n\n$molecule\nread\n$end\n\n@@@\n\n'.format(self.method.value,self.basis.value))
            f.write('$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n'
                    'basis {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n$molecule\nread\n$end\n'.format(self.method.value,self.basis.value))

    def run(self) -> None:
        ts_guess = self.ts_guess.receive()
        run_directory = self.run_directory.receive()
        
        #some calculators store charges this way
        charge = int(np.sum(ts_guess.get_initial_charges()))
        multiplicity = int(np.sum(ts_guess.get_initial_magnetic_moments()) + 1)
        filename = run_directory + "ts_guess.qcin"
        self._writetsqcin(
            structure = ts_guess,
            filename = filename,
            chg = charge,
            mult = multiplicity
        )
        os.system(f"qchem -nt {self.num_threads.value} {filename} {filename}.out")