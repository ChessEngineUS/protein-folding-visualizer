"""Main pipeline orchestrator for AlphaFold 3 and Boltz-2."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass

from ..alphafold3 import AlphaFold3Predictor, AF3Result
from ..boltz2 import Boltz2Predictor, Boltz2Result
from ..visualization import StructureViewer, ConfidencePlotter


@dataclass
class PipelineResult:
    """Container for complete pipeline results."""
    af3_result: Optional[AF3Result]
    boltz2_results: List[Boltz2Result]
    execution_time: float
    report_path: Optional[str]
    metadata: Dict


class ProteinPipeline:
    """
    Unified pipeline for protein structure prediction and analysis.
    
    Combines AlphaFold 3 and Boltz-2 for comprehensive structure
    and affinity predictions.
    """
    
    def __init__(
        self,
        alphafold3_weights: str = './models/alphafold3',
        boltz2_weights: str = './models/boltz2',
        use_gpu: bool = True,
        output_base: str = './data/outputs'
    ):
        self.af3_predictor = AlphaFold3Predictor(
            model_dir=alphafold3_weights,
            use_gpu=use_gpu
        )
        
        self.boltz2_predictor = Boltz2Predictor(
            model_dir=boltz2_weights,
            use_gpu=use_gpu
        )
        
        self.viewer = StructureViewer()
        self.plotter = ConfidencePlotter()
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        sequence: str,
        ligands: Optional[List[str]] = None,
        run_alphafold: bool = True,
        run_boltz: bool = True,
        generate_report: bool = True,
        experiment_name: Optional[str] = None
    ) -> PipelineResult:
        """
        Run complete prediction pipeline.
        
        Args:
            sequence: Protein sequence
            ligands: List of ligand SMILES strings
            run_alphafold: Whether to run AlphaFold 3
            run_boltz: Whether to run Boltz-2
            generate_report: Whether to generate HTML report
            experiment_name: Name for this experiment
            
        Returns:
            PipelineResult with all predictions
        """
        start_time = time.time()
        
        # Create experiment directory
        if experiment_name is None:
            experiment_name = f"experiment_{int(time.time())}"
        
        exp_dir = self.output_base / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save input data
        fasta_path = exp_dir / 'input.fasta'
        with open(fasta_path, 'w') as f:
            f.write(f">protein\n{sequence}\n")
        
        # Run AlphaFold 3
        af3_result = None
        if run_alphafold:
            print("Running AlphaFold 3...")
            af3_dir = exp_dir / 'alphafold3'
            af3_result = self.af3_predictor.predict(
                fasta_path=str(fasta_path),
                output_dir=str(af3_dir)
            )
            
            # Generate confidence plots
            self.plotter.plot_plddt(
                af3_result.plddt,
                save_path=str(af3_dir / 'plddt.png')
            )
            
            self.plotter.plot_pae(
                af3_result.pae,
                save_path=str(af3_dir / 'pae.png')
            )
        
        # Run Boltz-2 for each ligand
        boltz2_results = []
        if run_boltz and ligands:
            for i, ligand_smiles in enumerate(ligands):
                print(f"Running Boltz-2 for ligand {i+1}/{len(ligands)}...")
                boltz_dir = exp_dir / f'boltz2_ligand_{i+1}'
                
                result = self.boltz2_predictor.predict_with_affinity(
                    protein_fasta=str(fasta_path),
                    ligand_smiles=ligand_smiles,
                    output_dir=str(boltz_dir)
                )
                
                boltz2_results.append(result)
                
                # Generate plots
                if result.plddt is not None:
                    self.plotter.plot_plddt(
                        result.plddt,
                        save_path=str(boltz_dir / 'plddt.png')
                    )
        
        execution_time = time.time() - start_time
        
        # Generate report
        report_path = None
        if generate_report:
            report_path = self._generate_report(
                exp_dir,
                af3_result,
                boltz2_results,
                execution_time
            )
        
        # Save metadata
        metadata = {
            'experiment_name': experiment_name,
            'sequence': sequence,
            'ligands': ligands,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        metadata_path = exp_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return PipelineResult(
            af3_result=af3_result,
            boltz2_results=boltz2_results,
            execution_time=execution_time,
            report_path=report_path,
            metadata=metadata
        )
    
    def _generate_report(
        self,
        exp_dir: Path,
        af3_result: Optional[AF3Result],
        boltz2_results: List[Boltz2Result],
        execution_time: float
    ) -> str:
        """
        Generate HTML report with all results.
        
        Args:
            exp_dir: Experiment directory
            af3_result: AlphaFold 3 result
            boltz2_results: List of Boltz-2 results
            execution_time: Total execution time
            
        Returns:
            Path to generated report
        """
        report_path = exp_dir / 'report.html'
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Protein Folding Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; }
        .section { margin: 20px 0; padding: 15px; background: #ecf0f1; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Protein Structure Prediction Report</h1>
"""
        
        # Execution summary
        html += f"""<div class="section">
    <h2>Execution Summary</h2>
    <div class="metric"><strong>Total Time:</strong> {execution_time:.2f}s</div>
</div>
"""
        
        # AlphaFold 3 results
        if af3_result:
            html += f"""<div class="section">
    <h2>AlphaFold 3 Results</h2>
    <div class="metric"><strong>Mean pLDDT:</strong> {af3_result.plddt.mean():.2f}</div>
    <div class="metric"><strong>pTM:</strong> {af3_result.ptm:.3f}</div>
"""
            if af3_result.iptm:
                html += f'<div class="metric"><strong>ipTM:</strong> {af3_result.iptm:.3f}</div>'
            
            html += """<h3>Confidence Plots</h3>
    <img src="alphafold3/plddt.png" alt="pLDDT">
    <img src="alphafold3/pae.png" alt="PAE">
</div>
"""
        
        # Boltz-2 results
        if boltz2_results:
            html += '<div class="section"><h2>Boltz-2 Affinity Predictions</h2>'
            
            for i, result in enumerate(boltz2_results, 1):
                html += f"""<h3>Ligand {i}</h3>
<div class="metric"><strong>IC50:</strong> {result.ic50:.2f} nM</div>
<div class="metric"><strong>ΔG:</strong> {result.delta_g:.2f} kcal/mol</div>
<div class="metric"><strong>Binding Probability:</strong> {result.binding_probability:.2%}</div>
<img src="boltz2_ligand_{i}/plddt.png" alt="Boltz-2 pLDDT">
"""
            
            html += '</div>'
        
        html += """</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        return str(report_path)
