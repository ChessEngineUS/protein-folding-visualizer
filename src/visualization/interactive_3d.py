"""Enhanced interactive 3D visualization with NGLView."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    import nglview as nv
    NGL_AVAILABLE = True
except ImportError:
    NGL_AVAILABLE = False


class Interactive3DViewer:
    """Advanced interactive 3D visualization using NGLView."""
    
    def __init__(self):
        if not NGL_AVAILABLE:
            raise ImportError(
                "NGLView is not installed. Install with: pip install nglview"
            )
    
    def show_structure(
        self,
        pdb_path: str,
        representation: str = 'cartoon',
        color_scheme: str = 'chainid'
    ):
        """
        Display structure with NGLView.
        
        Args:
            pdb_path: Path to PDB file
            representation: Representation type
            color_scheme: Color scheme
            
        Returns:
            NGLView widget
        """
        view = nv.show_file(pdb_path)
        view.clear_representations()
        view.add_representation(representation, color_scheme=color_scheme)
        return view
    
    def show_complex(
        self,
        pdb_path: str,
        protein_selection: str = 'protein',
        ligand_selection: str = 'hetero'
    ):
        """
        Display protein-ligand complex.
        
        Args:
            pdb_path: PDB file path
            protein_selection: Selection for protein
            ligand_selection: Selection for ligand
            
        Returns:
            NGLView widget
        """
        view = nv.show_file(pdb_path)
        view.clear_representations()
        
        # Protein representation
        view.add_representation(
            'cartoon',
            selection=protein_selection,
            color='residueindex'
        )
        
        # Ligand representation
        view.add_representation(
            'ball+stick',
            selection=ligand_selection,
            color='element'
        )
        
        # Add surface
        view.add_representation(
            'surface',
            selection=protein_selection,
            opacity=0.3,
            color='hydrophobicity'
        )
        
        return view
    
    def show_with_confidence(
        self,
        pdb_path: str,
        plddt: np.ndarray
    ):
        """
        Display structure colored by confidence.
        
        Args:
            pdb_path: PDB file path
            plddt: pLDDT scores
            
        Returns:
            NGLView widget
        """
        view = nv.show_file(pdb_path)
        view.clear_representations()
        
        # Create custom color scheme based on pLDDT
        color_list = []
        for score in plddt:
            if score > 90:
                color_list.append([0, 83, 214])  # Blue
            elif score > 70:
                color_list.append([101, 203, 243])  # Cyan
            elif score > 50:
                color_list.append([255, 219, 19])  # Yellow
            else:
                color_list.append([255, 125, 69])  # Orange
        
        view.add_representation(
            'cartoon',
            color=color_list
        )
        
        return view
    
    def create_animation(
        self,
        pdb_paths: List[str],
        delay: int = 500
    ):
        """
        Create animation from multiple structures.
        
        Args:
            pdb_paths: List of PDB file paths
            delay: Delay between frames (ms)
            
        Returns:
            NGLView widget with animation
        """
        view = nv.show_file(pdb_paths[0])
        
        for pdb_path in pdb_paths[1:]:
            view.add_trajectory(nv.FileStructure(pdb_path))
        
        view.player.delay = delay
        
        return view
