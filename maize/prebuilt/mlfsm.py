from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node
import os
from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer
from mlfsm.geom import project_trans_rot
import numpy as np
import ase

class RunMLFSM(Node):
    """
    Run the ML-FSM.
    
    Inputs:
        -Reactant: ase.Atoms
        -Product: ase.Atoms
        -Calculator: ase.calculator
        -run_directory: str
    Outputs:
        -TS Guess: ase.Atoms
    """
    reactant = Input["ase.Atoms"]()
    product = Input["ase.Atoms"]()
    calculator = Input["ASECalculator"]()
    run_directory = Output[str]()
    ts_out = Output[str]()
        
    nnodes_min: Parameter[int] = Parameter(default=18)
    interp: Parameter[str] = Parameter(default="ric")
    ninterp: Parameter[int] = Parameter(default=50)
    method: Parameter[str] = Parameter(default="L-BFGS-B")
    maxls: Parameter[int] = Parameter(default=3)
    maxiter: Parameter[int] = Parameter(default=2)
    dmax: Parameter[float] = Parameter(default=0.05)
    outdir: Parameter[str] = Parameter(default=".")
    
    def get_ts(self, fsm_string) -> ase.Atoms:
        """
        Get the ase.Atoms object of the TS guess from final FSM string
        """
        path = fsm_string.r_string + fsm_string.p_string[::-1]
        energy = np.array(fsm_string.r_energy + fsm_string.p_energy[::-1])
        energy = list(energy - energy.min())
        ts_index = energy.index(max(energy))
        return path[ts_index]
        
    
    def run(self) -> None:
        """
        Runs FSM with specified parameters and inputs
        """

        #Get inputs
        reactant = self.reactant.receive()
        product = self.product.receive()
        calculator = self.calculator.receive()
        
        #Align reactant and product
        _,prod_aligned = project_trans_rot(reactant.get_positions(),product.get_positions())
        product.set_positions(prod_aligned.reshape(-1,3))
        
        #Set up optimizer
        optimizer = CartesianOptimizer(
            calculator,
            self.method.value,
            self.maxiter.value,
            self.maxls.value,
            self.dmax.value
        )

        #Build the string
        string = FreezingString(
            reactant,
            product,
            self.nnodes_min.value,
            self.interp.value,
            self.ninterp.value,
        )

        #Run FSM
        while string.growing:
            string.grow()
            string.optimize(optimizer)
            string.write(self.outdir.value)

        #get TS guess geometry
        ts_guess = self.get_ts(string)

        #send TS guess
        self.ts_out.send(ts_guess)
        self.run_directory.send(self.outdir.value)                  
