"""Utility functions for Boltz-2 predictions."""

from typing import Dict, List, Optional
from pathlib import Path
import json


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # If RDKit not available, do basic check
        return len(smiles) > 0 and smiles.isprintable()


def smiles_to_mol_properties(smiles: str) -> Dict:
    """
    Extract molecular properties from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular properties
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
        }
    except ImportError:
        return {'smiles': smiles}


def prepare_ligand_input(
    smiles: str,
    output_path: str,
    ligand_id: str = 'LIG'
) -> str:
    """
    Prepare ligand input file for Boltz-2.
    
    Args:
        smiles: SMILES string
        output_path: Output file path
        ligand_id: Ligand identifier
        
    Returns:
        Path to created file
    """
    ligand_data = {
        'id': ligand_id,
        'smiles': smiles,
        'properties': smiles_to_mol_properties(smiles)
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ligand_data, f, indent=2)
    
    return output_path
