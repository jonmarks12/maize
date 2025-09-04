from pydantic import BaseModel, validator
from typing import List, Optional
import numpy as np
from ase import Atoms

class FSMInput(BaseModel):
    """validate fsm inputs"""
    
    @staticmethod
    def validate_fsm_input(reactant,product) -> None:
        r_symbols = reactant.get_chemical_symbols()
        p_symbols = product.get_chemical_symbols()
        
        if len(r_symbols) != len(p_symbols):
            raise ValueError(
                f"ML-FSM requires equal number of atoms in reactant and product"
                f"reactant: {len(r_symbols)}, product={len(p_symbols)}"
            )
        if r_symbols == p_symbols:
            raise ValueError(
                f"ML-FSM requires same identical atom ordering in reactant and product"
                f"Reactant: {r_symbols}"
                f"Product: {p_symbols}"
            )
            