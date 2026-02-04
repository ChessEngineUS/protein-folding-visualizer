"""Binding affinity analysis tools for Boltz-2."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AffinityResult:
    """Container for binding affinity analysis."""
    ic50: float  # nM
    delta_g: float  # kcal/mol
    ki: Optional[float]  # nM
    binding_probability: float
    confidence_interval: Tuple[float, float]
    metadata: Dict


class AffinityCalculator:
    """Tools for analyzing binding affinity predictions."""
    
    # Physical constants
    R = 1.987e-3  # kcal/(mol*K)
    T = 298.15  # K (25°C)
    
    @staticmethod
    def delta_g_to_kd(delta_g: float) -> float:
        """
        Convert ΔG to dissociation constant Kd.
        
        Args:
            delta_g: Binding free energy in kcal/mol
            
        Returns:
            Kd in nM
        """
        RT = AffinityCalculator.R * AffinityCalculator.T
        kd_M = np.exp(delta_g / RT)
        return kd_M * 1e9  # Convert M to nM
    
    @staticmethod
    def kd_to_delta_g(kd_nm: float) -> float:
        """
        Convert dissociation constant to ΔG.
        
        Args:
            kd_nm: Kd in nM
            
        Returns:
            ΔG in kcal/mol
        """
        RT = AffinityCalculator.R * AffinityCalculator.T
        kd_M = kd_nm * 1e-9
        return RT * np.log(kd_M)
    
    @staticmethod
    def ic50_to_ki(ic50_nm: float, substrate_conc: float, km: float) -> float:
        """
        Convert IC50 to inhibition constant Ki using Cheng-Prusoff equation.
        
        Args:
            ic50_nm: IC50 in nM
            substrate_conc: Substrate concentration in nM
            km: Michaelis constant in nM
            
        Returns:
            Ki in nM
        """
        return ic50_nm / (1 + substrate_conc / km)
    
    @staticmethod
    def parse_results(result) -> AffinityResult:
        """
        Parse Boltz-2 affinity prediction results.
        
        Args:
            result: Boltz2Result object
            
        Returns:
            AffinityResult with processed metrics
        """
        if result.ic50 is None or result.delta_g is None:
            raise ValueError("Affinity data not available in result")
        
        # Calculate Ki (assuming competitive inhibition)
        # Default values if not provided
        ki = result.ic50  # Simplified; adjust with substrate info
        
        # Calculate confidence interval (±1 kcal/mol typical uncertainty)
        dg_lower = result.delta_g - 1.0
        dg_upper = result.delta_g + 1.0
        
        return AffinityResult(
            ic50=result.ic50,
            delta_g=result.delta_g,
            ki=ki,
            binding_probability=result.binding_probability or 0.5,
            confidence_interval=(dg_lower, dg_upper),
            metadata=result.metadata
        )
    
    @staticmethod
    def classify_affinity(ic50_nm: float) -> str:
        """
        Classify binding affinity strength.
        
        Args:
            ic50_nm: IC50 in nM
            
        Returns:
            Classification string
        """
        if ic50_nm < 1:
            return "Very Strong (sub-nM)"
        elif ic50_nm < 10:
            return "Strong (low nM)"
        elif ic50_nm < 100:
            return "Moderate (nM)"
        elif ic50_nm < 1000:
            return "Weak (high nM)"
        elif ic50_nm < 10000:
            return "Very Weak (μM)"
        else:
            return "Non-binding (>10 μM)"
    
    @staticmethod
    def calculate_efficiency_metrics(
        delta_g: float,
        molecular_weight: float,
        num_heavy_atoms: int
    ) -> Dict[str, float]:
        """
        Calculate ligand efficiency metrics.
        
        Args:
            delta_g: Binding free energy (kcal/mol)
            molecular_weight: Molecular weight (Da)
            num_heavy_atoms: Number of heavy atoms
            
        Returns:
            Dictionary of efficiency metrics
        """
        # Ligand Efficiency (LE)
        le = -delta_g / num_heavy_atoms if num_heavy_atoms > 0 else 0
        
        # Size-independent Ligand Efficiency (SILE)
        sile = -delta_g / num_heavy_atoms**0.3 if num_heavy_atoms > 0 else 0
        
        # Lipophilic Ligand Efficiency (LLE) - requires logP
        # Placeholder: lle = -delta_g - logP
        
        return {
            'ligand_efficiency': le,
            'sile': sile,
            'binding_energy_per_atom': le
        }
