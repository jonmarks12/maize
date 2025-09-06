from maize.core.interface import Parameter, Output, Input, MultiOutput
from maize.core.node import Node

class OptimizeGeometryAtoms(Node):
    """
    Receive ASE Atoms and return optimized ASE Atoms.
    """
    atoms_in: Input["ASEAtoms"] = Input()
    method: Parameter[str] = Parameter(default="uma")
    fmax: Parameter[float] = Parameter(default=0.05)
    workdir: Parameter[str] = Parameter(default="work_opt")
    atoms_out: Output["ASEAtoms"] = Output()
    def run(self) -> None:
        from ase.io import read, write  # lazy import
        atoms = self.atoms_in.receive()
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]
        from ase.optimize import FIRE

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")

        # Use node parameter (default 0.05 eV/Å)
        fmax = float(self.fmax.value)

        def _optimize(f_max: float, structure, calc):
            structure = structure.copy()
            structure.calc = calc
            dyn = FIRE(structure)
            dyn.run(fmax=f_max)
            return structure

        optimized = _optimize(fmax, atoms, calc)
        self.atoms_out.send(optimized)