"""AlphaFold 3 prediction wrapper with visualization integration."""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class AF3Result:
    """Container for AlphaFold 3 prediction results."""
    pdb_path: str
    plddt: np.ndarray
    pae: np.ndarray
    ptm: float
    iptm: Optional[float]
    ranking_confidence: float
    metadata: Dict


class AlphaFold3Predictor:
    """
    AlphaFold 3 prediction interface.
    
    Uses the official DeepMind implementation with diffusion-based
    structure generation.
    """
    
    def __init__(
        self,
        model_dir: str = './models/alphafold3',
        database_dir: Optional[str] = None,
        use_gpu: bool = True,
        num_gpus: int = 1
    ):
        self.model_dir = Path(model_dir)
        self.database_dir = Path(database_dir) if database_dir else None
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus
        
        self._validate_installation()
    
    def _validate_installation(self):
        """Check if AlphaFold 3 is properly installed."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"AlphaFold 3 model directory not found: {self.model_dir}"
            )
        
        required_files = ['model.pth', 'config.json']
        for fname in required_files:
            if not (self.model_dir / fname).exists():
                raise FileNotFoundError(
                    f"Required file not found: {fname}"
                )
    
    def predict(
        self,
        fasta_path: str,
        output_dir: str,
        num_recycling: int = 3,
        num_diffusion_steps: int = 200,
        use_templates: bool = True,
        msa_mode: str = 'full',
        random_seed: Optional[int] = None
    ) -> AF3Result:
        """
        Run AlphaFold 3 structure prediction.
        
        Args:
            fasta_path: Path to input FASTA file
            output_dir: Directory for outputs
            num_recycling: Number of recycling iterations
            num_diffusion_steps: Diffusion denoising steps
            use_templates: Whether to use template structures
            msa_mode: MSA generation mode ('full', 'reduced', 'none')
            random_seed: Random seed for reproducibility
            
        Returns:
            AF3Result object with predictions and metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare configuration
        config = {
            'fasta_path': str(fasta_path),
            'output_dir': str(output_dir),
            'model_dir': str(self.model_dir),
            'num_recycling': num_recycling,
            'num_diffusion_steps': num_diffusion_steps,
            'use_templates': use_templates,
            'msa_mode': msa_mode,
            'use_gpu': self.use_gpu,
            'num_gpus': self.num_gpus
        }
        
        if random_seed is not None:
            config['random_seed'] = random_seed
        
        if self.database_dir:
            config['database_dir'] = str(self.database_dir)
        
        # Write config file
        config_path = output_path / 'af3_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run AlphaFold 3
        cmd = [
            'python', '-m', 'alphafold3.run',
            f'--config_path={config_path}'
        ]
        
        print(f"Running AlphaFold 3 prediction...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.model_dir.parent
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"AlphaFold 3 failed:\n{result.stderr}"
            )
        
        # Parse results
        return self._parse_results(output_path)
    
    def _parse_results(self, output_dir: Path) -> AF3Result:
        """Parse AlphaFold 3 output files."""
        # Find best ranked model
        ranking_file = output_dir / 'ranking_debug.json'
        with open(ranking_file) as f:
            ranking = json.load(f)
        
        best_model = ranking['order'][0]
        
        # Load structure
        pdb_path = output_dir / f'ranked_{best_model}.pdb'
        
        # Load confidence metrics
        confidence_file = output_dir / f'confidence_{best_model}.json'
        with open(confidence_file) as f:
            confidence = json.load(f)
        
        plddt = np.array(confidence['plddt'])
        pae = np.array(confidence['pae'])
        ptm = confidence['ptm']
        iptm = confidence.get('iptm')
        ranking_confidence = confidence['ranking_confidence']
        
        return AF3Result(
            pdb_path=str(pdb_path),
            plddt=plddt,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            ranking_confidence=ranking_confidence,
            metadata=confidence
        )
    
    def predict_complex(
        self,
        protein_fasta: str,
        ligand_data: Optional[str] = None,
        dna_fasta: Optional[str] = None,
        rna_fasta: Optional[str] = None,
        output_dir: str = 'output',
        **kwargs
    ) -> AF3Result:
        """
        Predict biomolecular complex structure.
        
        Supports protein-ligand, protein-DNA, protein-RNA complexes.
        """
        # Create combined input JSON
        input_data = {'protein': protein_fasta}
        
        if ligand_data:
            input_data['ligand'] = ligand_data
        if dna_fasta:
            input_data['dna'] = dna_fasta
        if rna_fasta:
            input_data['rna'] = rna_fasta
        
        input_path = Path(output_dir) / 'complex_input.json'
        input_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'w') as f:
            json.dump(input_data, f)
        
        return self.predict(
            fasta_path=str(input_path),
            output_dir=output_dir,
            **kwargs
        )
