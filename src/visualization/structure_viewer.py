"""Interactive 3D structure visualization."""

import py3Dmol
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class StructureViewer:
    """Interactive 3D protein structure visualization."""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
    
    def show_structure(
        self,
        pdb_path: str,
        confidence: Optional[np.ndarray] = None,
        style: str = 'cartoon',
        color_by_confidence: bool = True
    ) -> py3Dmol.view:
        """
        Display protein structure in interactive 3D viewer.
        
        Args:
            pdb_path: Path to PDB file
            confidence: pLDDT scores for coloring
            style: Visualization style ('cartoon', 'stick', 'sphere')
            color_by_confidence: Color by confidence scores
            
        Returns:
            py3Dmol view object
        """
        view = py3Dmol.view(width=self.width, height=self.height)
        
        with open(pdb_path) as f:
            pdb_data = f.read()
        
        view.addModel(pdb_data, 'pdb')
        
        if color_by_confidence and confidence is not None:
            # Color by confidence: blue (high) to red (low)
            self._color_by_plddt(view, confidence)
        else:
            view.setStyle({style: {'color': 'spectrum'}})
        
        view.zoomTo()
        return view
    
    def show_complex(
        self,
        pdb_path: str,
        protein_chain: str = 'A',
        ligand_chain: Optional[str] = 'B',
        confidence: Optional[np.ndarray] = None
    ) -> py3Dmol.view:
        """
        Visualize protein-ligand complex.
        
        Args:
            pdb_path: PDB file path
            protein_chain: Protein chain ID
            ligand_chain: Ligand chain ID
            confidence: Confidence scores
            
        Returns:
            py3Dmol view
        """
        view = py3Dmol.view(width=self.width, height=self.height)
        
        with open(pdb_path) as f:
            pdb_data = f.read()
        
        view.addModel(pdb_data, 'pdb')
        
        # Style protein
        view.setStyle(
            {'chain': protein_chain},
            {'cartoon': {'color': 'spectrum'}}
        )
        
        # Style ligand
        if ligand_chain:
            view.setStyle(
                {'chain': ligand_chain},
                {'stick': {'colorscheme': 'greenCarbon'}}
            )
        
        # Add surface
        view.addSurface(
            py3Dmol.VDW,
            {'opacity': 0.3, 'color': 'white'},
            {'chain': protein_chain}
        )
        
        view.zoomTo()
        return view
    
    def _color_by_plddt(
        self,
        view: py3Dmol.view,
        plddt: np.ndarray
    ):
        """Apply pLDDT-based coloring scheme."""
        # AlphaFold coloring:
        # Very high (pLDDT > 90): blue
        # Confident (90 > pLDDT > 70): cyan
        # Low (70 > pLDDT > 50): yellow
        # Very low (pLDDT < 50): orange/red
        
        for i, score in enumerate(plddt):
            if score > 90:
                color = 'blue'
            elif score > 70:
                color = 'cyan'
            elif score > 50:
                color = 'yellow'
            else:
                color = 'orange'
            
            view.setStyle(
                {'resi': i + 1},
                {'cartoon': {'color': color}}
            )
    
    def create_comparison_view(
        self,
        pdb_paths: List[str],
        labels: List[str]
    ) -> py3Dmol.view:
        """
        Create side-by-side comparison of multiple structures.
        
        Args:
            pdb_paths: List of PDB file paths
            labels: Labels for each structure
            
        Returns:
            py3Dmol view with aligned structures
        """
        view = py3Dmol.view(
            width=self.width,
            height=self.height,
            viewergrid=(1, len(pdb_paths))
        )
        
        for idx, (pdb_path, label) in enumerate(zip(pdb_paths, labels)):
            with open(pdb_path) as f:
                pdb_data = f.read()
            
            view.addModel(pdb_data, 'pdb', viewer=(0, idx))
            view.setStyle(
                {'cartoon': {'color': 'spectrum'}},
                viewer=(0, idx)
            )
            view.zoomTo(viewer=(0, idx))
        
        return view
