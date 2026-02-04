"""Structural and affinity evaluation metrics."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class StructureMetrics:
    """Container for structure quality metrics."""
    ramachandran_favored: float  # % in favored regions
    ramachandran_outliers: float  # % outliers
    clash_score: float
    molprobity_score: float
    rotamer_outliers: float  # %


@dataclass
class AffinityMetrics:
    """Container for binding affinity metrics."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r_squared: float  # R² correlation
    pearson_r: float  # Pearson correlation
    spearman_rho: float  # Spearman rank correlation


class MetricsCalculator:
    """Calculate advanced evaluation metrics."""
    
    @staticmethod
    def calculate_contact_precision(
        predicted_contacts: np.ndarray,
        native_contacts: np.ndarray,
        top_k: Optional[int] = None
    ) -> float:
        """
        Calculate contact prediction precision.
        
        Args:
            predicted_contacts: Predicted contact map
            native_contacts: Native contact map
            top_k: Consider only top-k predictions
            
        Returns:
            Precision score
        """
        if top_k:
            # Get indices of top-k predictions
            flat_pred = predicted_contacts.flatten()
            top_indices = np.argsort(flat_pred)[-top_k:]
            
            pred_binary = np.zeros_like(flat_pred)
            pred_binary[top_indices] = 1
            pred_binary = pred_binary.reshape(predicted_contacts.shape)
        else:
            pred_binary = (predicted_contacts > 0.5).astype(int)
        
        native_binary = (native_contacts > 0).astype(int)
        
        tp = np.sum(pred_binary * native_binary)
        fp = np.sum(pred_binary * (1 - native_binary))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precision
    
    @staticmethod
    def calculate_secondary_structure_accuracy(
        predicted_ss: str,
        native_ss: str
    ) -> Dict[str, float]:
        """
        Calculate secondary structure prediction accuracy.
        
        Args:
            predicted_ss: Predicted SS string (H, E, C)
            native_ss: Native SS string
            
        Returns:
            Dictionary of accuracy metrics
        """
        if len(predicted_ss) != len(native_ss):
            raise ValueError("Sequence lengths must match")
        
        total = len(predicted_ss)
        correct = sum(p == n for p, n in zip(predicted_ss, native_ss))
        
        # Per-class accuracy
        ss_types = ['H', 'E', 'C']  # Helix, Sheet, Coil
        per_class = {}
        
        for ss_type in ss_types:
            mask = [n == ss_type for n in native_ss]
            if sum(mask) == 0:
                per_class[ss_type] = 0.0
                continue
            
            correct_class = sum(
                p == n for p, n, m in zip(predicted_ss, native_ss, mask) if m
            )
            per_class[ss_type] = correct_class / sum(mask)
        
        return {
            'overall_accuracy': correct / total,
            'helix_accuracy': per_class.get('H', 0.0),
            'sheet_accuracy': per_class.get('E', 0.0),
            'coil_accuracy': per_class.get('C', 0.0)
        }
    
    @staticmethod
    def calculate_affinity_metrics(
        predicted: np.ndarray,
        experimental: np.ndarray
    ) -> AffinityMetrics:
        """
        Calculate binding affinity prediction metrics.
        
        Args:
            predicted: Predicted affinities (e.g., pKd or ΔG)
            experimental: Experimental values
            
        Returns:
            AffinityMetrics object
        """
        # Remove NaN values
        mask = ~(np.isnan(predicted) | np.isnan(experimental))
        pred = predicted[mask]
        exp = experimental[mask]
        
        if len(pred) == 0:
            return AffinityMetrics(0, 0, 0, 0, 0)
        
        # Calculate metrics
        mae = np.mean(np.abs(pred - exp))
        rmse = np.sqrt(np.mean((pred - exp) ** 2))
        
        # R²
        ss_res = np.sum((exp - pred) ** 2)
        ss_tot = np.sum((exp - np.mean(exp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlations
        pearson_r, _ = stats.pearsonr(pred, exp)
        spearman_rho, _ = stats.spearmanr(pred, exp)
        
        return AffinityMetrics(
            mae=mae,
            rmse=rmse,
            r_squared=r_squared,
            pearson_r=pearson_r,
            spearman_rho=spearman_rho
        )
    
    @staticmethod
    def calculate_interface_metrics(
        complex_pdb: str,
        chain_a: str = 'A',
        chain_b: str = 'B'
    ) -> Dict[str, float]:
        """
        Calculate protein-protein interface metrics.
        
        Args:
            complex_pdb: Path to complex structure
            chain_a: First chain ID
            chain_b: Second chain ID
            
        Returns:
            Dictionary of interface metrics
        """
        from Bio.PDB import PDBParser, NeighborSearch
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', complex_pdb)
        
        # Get atoms from each chain
        chain_a_atoms = [atom for atom in structure[0][chain_a].get_atoms()]
        chain_b_atoms = [atom for atom in structure[0][chain_b].get_atoms()]
        
        # Find interface contacts
        ns = NeighborSearch(chain_a_atoms + chain_b_atoms)
        interface_contacts = 0
        
        for atom_a in chain_a_atoms:
            nearby = ns.search(atom_a.coord, 5.0)  # 5Å cutoff
            for atom_b in nearby:
                if atom_b in chain_b_atoms:
                    interface_contacts += 1
        
        # Calculate interface area (simplified)
        interface_area = interface_contacts * 20  # Rough estimate
        
        return {
            'interface_contacts': interface_contacts,
            'interface_area': interface_area,
            'normalized_interface': interface_contacts / len(chain_a_atoms)
        }
