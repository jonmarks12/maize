from typing import Optional, Dict, Any
import os
from maize.core.interface import Parameter, Output, Input, MultiOutput
from maize.core.node import Node
from maize.core.graph import Graph
from maize.prebuilt.mlfsm import RunMLFSM
from maize.prebuilt.qchem_prfo import RunPRFO
from maize.prebuilt.sella import RunSellaTS
from maize.prebuilt.ase_optimize import OptimizeGeometryAtoms
from maize.prebuilt.util_nodes import _FeedCalculator, _FeedAtoms, _FeedInitial
from maize.steps.plumbing import Merge, Copy


class TerminalNode(Node):
    inp = Input()

    def run(self) -> None:
        received = self.inp.receive()
        print("Reached Terminal Node")
        print(received)


class TS_refinement_controller(Node):
    ts_guess = Input["ASEAtoms"]()
    run_directory = Input[str]()
    run_mlfsm = Input[Dict[str, Any]]()
    ts_copy1 = Output["ASEAtoms"]()
    run_out1 = Output[str]()
    ts_copy2 = Output["ASEAtoms"]()
    run_out2 = Output[str]()

    def run(self) -> None:
        ts_guess = self.ts_guess.receive()
        run_dir = self.run_directory.receive()
        self.ts_copy1.send(ts_guess)
        self.run_out1.send(run_dir)
        run_mlfsm_guess = self.run_mlfsm.receive()
        if not run_mlfsm_guess["Success"]:
            self.ts_copy2.send(ts_guess)
            self.run_out2.send(run_dir)


class TSRefinement(Graph):
    ts_guess = Input["ASEAtoms"]()
    calculator = Input["ASECalculator"]()
    run_directory = Input[str]()

    out: Output[Dict[str, Any]]

    def build(self) -> None:
        controller = flow.add(TS_refinement_controller, name="duplicate_ts_guess")
        sella = flow.add(RunSellaTS, name="run_sella")
        prfo_sella = flow.add(
            RunPRFO, name="run_prfo_sella", parameters=dict(method="B3LYP", basis="6-31g(d)")
        )
        prfo_fsm = flow.add(
            RunPRFO, name="run_prfo_fsm", parameters=dict(method="B3LYP", basis="6-31g(d)")
        )
        merge = flow.add(Merge[Dict[str, Any]])

        # map ports to appropriate nodes in SG
        self.ts_guess = self.map_port(controller.ts_guess)
        self.calculator = self.map_port(sella.calculator)
        self.run_directory = self.map_port(controller.run_directory)

        self.connect(controller.ts_copy1, sella.ts_guess)
        self.connect(controller.run_out1, sella.run_directory)
        self.connect(sella.ts_out_atoms, prfo_sella.ts_guess)
        self.connect(sella.ts_out_loc, prfo_sella.run_directory)
        self.connect(prfo_sella.output_failed, controller.run_mlfsm)
        self.connect(prfo_sella.output_success, merge.inp)
        self.connect(controller.ts_copy2, prfo_fsm.ts_guess)
        self.connect(controller.run_out2, prfo_fsm.run_directory)
        self.connect(prfo_fsm.output_success, merge.inp)
        self.connect(prfo_fsm.output_failed, merge.inp)

        self.out = self.map_port(merge.out)


# build and run
if __name__ == "__main__":
    import argparse, os
    from maize.core.workflow import Workflow

    parser = argparse.ArgumentParser(description="MAIZE TS workflow with UMA-m and ML-FSM")
    parser.add_argument("--initial", required=True, help="Path to initial.xyz containing R/P")
    parser.add_argument(
        "--ts-out", default="ts_guess.xyz", help="Filename for TS guess (written in ts workdir)"
    )

    # OptimizeGeometryAtoms
    parser.add_argument("--fmax", type=float, default=0.05, help="FIRE force threshold (eV/Å)")

    # RunMLFSM controls
    parser.add_argument("--interp", choices=["ric", "lst", "cart"], default="ric")
    parser.add_argument("--calculator", choices=["uma_s", "uma_m", "eSEN"], default="uma_m")
    parser.add_argument("--nnodes-min", type=int, default=18)
    parser.add_argument("--ninterp", type=int, default=50)
    parser.add_argument("--method", choices=["L-BFGS-B", "CG"], default="L-BFGS-B")
    parser.add_argument("--maxls", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--dmax", type=float, default=0.05)
    parser.add_argument("--interpolate-only", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = os.path.dirname(args.initial)

    # Build the graph
    flow = Workflow(name="ts_search_with_mlfsm")
    feedCalc = flow.add(
        _FeedCalculator, name="feed_calculator", parameters=dict(calculator=args.calculator)
    )
    feedI = flow.add(_FeedInitial, name="feed_initial", parameters=dict(path=args.initial))
    copy = flow.add(Copy["ASECalculator"])

    optR = flow.add(
        OptimizeGeometryAtoms,
        name="opt_reactant",
        parameters=dict(
            fmax=args.fmax,
            workdir=os.path.join(args.outdir, "opt_reactant"),
        ),
    )
    optP = flow.add(
        OptimizeGeometryAtoms,
        name="opt_product",
        parameters=dict(
            fmax=args.fmax,
            workdir=os.path.join(args.outdir, "opt_product"),
        ),
    )

    mlfsm = flow.add(
        RunMLFSM,
        name="mlfsm_run",
        parameters=dict(
            interp=args.interp,
            nnodes_min=args.nnodes_min,
            ninterp=args.ninterp,
            method=args.method,
            maxls=args.maxls,
            maxiter=args.maxiter,
            dmax=args.dmax,
            outdir=args.outdir,
        ),
    )

    SG = flow.add(TSRefinement, name="TS_Refinement")
    terminal_1 = flow.add(TerminalNode, name="Terminal Node1")

    # Wire it up
    flow.connect(feedI.reactant, optR.atoms_in)
    flow.connect(feedI.product, optP.atoms_in)
    flow.connect(feedCalc.out, copy.inp)
    flow.connect(copy.out, mlfsm.calculator)
    flow.connect(copy.out, SG.calculator)
    flow.connect(optR.atoms_out, mlfsm.reactant)
    flow.connect(optP.atoms_out, mlfsm.product)
    flow.connect(mlfsm.ts_out, SG.ts_guess)
    flow.connect(mlfsm.fsm_loc, SG.run_directory)
    flow.connect(SG.out, terminal_1.inp)

    # Run
    flow.check()
    flow.execute()
