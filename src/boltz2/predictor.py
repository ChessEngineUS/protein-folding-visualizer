"""Boltz-2 prediction and affinity calculation."""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class Boltz2Result:
    """Container for Boltz-2 prediction results."""
    pdb_path: str
    plddt: np.ndarray
    pae: np.ndarray
    ic50: Optional[float]  # nM
    delta_g: Optional[float]  # kcal/mol
    binding_probability: Optional[float]
    metadata: Dict


class Boltz2Predictor:
    """
    Boltz-2 predictor for structure and binding affinity.
    
    1000x faster than FEP methods while approaching FEP accuracy.
    """
    
    def __init__(
        self,
        model_dir: str = './models/boltz2',
        use_gpu: bool = True,
        device: str = 'cuda:0'
    ):
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.device = device
        
        self._validate_installation()
    
    def _validate_installation(self):
        """Check Boltz-2 installation."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Boltz-2 model directory not found: {self.model_dir}"
            )
    
    def predict(
        self,
        fasta_path: str,
        output_dir: str,
        num_recycling: int = 3,
        use_msa: bool = True,
        random_seed: Optional[int] = None
    ) -> Boltz2Result:
        """
        Run Boltz-2 structure prediction.
        
        Args:
            fasta_path: Input FASTA file
            output_dir: Output directory
            num_recycling: Number of recycling steps
            use_msa: Whether to fetch MSA data
            random_seed: Random seed
            
        Returns:
            Boltz2Result with structure and metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare config
        config = {
            'data': {'fasta_file': str(fasta_path)},
            'model': {
                'checkpoint': str(self.model_dir / 'boltz2.pth'),
                'recycling_steps': num_recycling,
                'use_msa': use_msa
            },
            'output': {'out_dir': str(output_dir)},
            'device': self.device if self.use_gpu else 'cpu'
        }
        
        if random_seed is not None:
            config['seed'] = random_seed
        
        config_path = output_path / 'boltz2_config.yaml'
        self._write_yaml(config, config_path)
        
        # Run Boltz-2
        cmd = [
            'boltz', 'predict',
            str(config_path)
        ]
        
        print(f"Running Boltz-2 prediction...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Boltz-2 failed:\n{result.stderr}")
        
        return self._parse_results(output_path)
    
    def predict_with_affinity(
        self,
        protein_fasta: str,
        ligand_smiles: str,
        output_dir: str,
        **kwargs
    ) -> Boltz2Result:
        """
        Predict structure and binding affinity for protein-ligand complex.
        
        Args:
            protein_fasta: Protein sequence file
            ligand_smiles: Ligand SMILES string
            output_dir: Output directory
            
        Returns:
            Boltz2Result with structure and affinity predictions
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create combined input
        input_data = {
            'sequences': [
                {'protein_chain': {'sequence_file': protein_fasta}},
                {'ligand': {'smiles': ligand_smiles}}
            ]
        }
        
        input_path = output_path / 'complex_input.json'
        with open(input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
        
        # Run prediction with affinity
        config = {
            'data': {'input_json': str(input_path)},
            'model': {
                'checkpoint': str(self.model_dir / 'boltz2.pth'),
                'predict_affinity': True
            },
            'output': {'out_dir': str(output_dir)},
            'device': self.device if self.use_gpu else 'cpu'
        }
        
        config_path = output_path / 'affinity_config.yaml'
        self._write_yaml(config, config_path)
        
        cmd = ['boltz', 'predict', str(config_path)]
        
        print(f"Running Boltz-2 affinity prediction...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Boltz-2 affinity prediction failed:\n{result.stderr}")
        
        return self._parse_results(output_path, has_affinity=True)
    
    def _parse_results(
        self,
        output_dir: Path,
        has_affinity: bool = False
    ) -> Boltz2Result:
        """Parse Boltz-2 output files."""
        # Find output PDB
        pdb_files = list(output_dir.glob('*.pdb'))
        if not pdb_files:
            raise FileNotFoundError("No PDB output found")
        
        pdb_path = pdb_files[0]
        
        # Load confidence scores
        confidence_file = output_dir / 'confidence.json'
        with open(confidence_file) as f:
            confidence = json.load(f)
        
        plddt = np.array(confidence['plddt'])
        pae = np.array(confidence['pae'])
        
        # Load affinity if available
        ic50, delta_g, binding_prob = None, None, None
        if has_affinity:
            affinity_file = output_dir / 'affinity.json'
            if affinity_file.exists():
                with open(affinity_file) as f:
                    affinity = json.load(f)
                
                ic50 = affinity.get('ic50_nm')
                delta_g = affinity.get('delta_g_kcal_mol')
                binding_prob = affinity.get('binding_probability')
        
        return Boltz2Result(
            pdb_path=str(pdb_path),
            plddt=plddt,
            pae=pae,
            ic50=ic50,
            delta_g=delta_g,
            binding_probability=binding_prob,
            metadata=confidence
        )
    
    @staticmethod
    def _write_yaml(data: Dict, path: Path):
        """Write YAML configuration file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(data, f)
