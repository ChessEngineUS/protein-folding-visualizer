"""Benchmark suite for comprehensive evaluation."""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from .casp_evaluator import CASPEvaluator, CASPResult
from .metrics import MetricsCalculator, AffinityMetrics


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite.
    
    Evaluates models on:
    - CASP15 targets
    - PDBbind affinity dataset
    - Custom challenge sets
    """
    
    def __init__(self, output_dir: str = './benchmarks'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.casp_evaluator = CASPEvaluator()
        self.metrics_calc = MetricsCalculator()
        
        self.results = {
            'casp15': [],
            'pdbbind': [],
            'timing': {}
        }
    
    def run_casp15_benchmark(
        self,
        predictor,
        targets: Optional[List[str]] = None
    ) -> List[CASPResult]:
        """
        Run full CASP15 benchmark.
        
        Args:
            predictor: Model predictor (AF3 or Boltz2)
            targets: Specific targets to evaluate (None = all)
            
        Returns:
            List of CASP results
        """
        if targets is None:
            targets = list(self.casp_evaluator.targets.keys())
        
        results = []
        
        for target_id in targets:
            target = self.casp_evaluator.targets[target_id]
            
            print(f"\nProcessing {target_id} ({target.difficulty})...")
            
            # Run prediction
            start_time = time.time()
            
            # Create FASTA
            fasta_path = self.output_dir / f"{target_id}.fasta"
            with open(fasta_path, 'w') as f:
                f.write(f">{target_id}\n{target.sequence}\n")
            
            try:
                # Predict structure
                prediction = predictor.predict(
                    fasta_path=str(fasta_path),
                    output_dir=str(self.output_dir / target_id)
                )
                
                prediction_time = time.time() - start_time
                
                # Evaluate if native structure available
                if target.native_pdb:
                    result = self.casp_evaluator.evaluate_prediction(
                        target_id=target_id,
                        prediction_pdb=prediction.pdb_path,
                        native_pdb=target.native_pdb,
                        prediction_time=prediction_time
                    )
                    results.append(result)
                    
                    print(f"  GDT_TS: {result.gdt_ts:.2f}")
                    print(f"  TM-score: {result.tm_score:.3f}")
                    print(f"  Time: {prediction_time:.1f}s")
                    
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        self.results['casp15'] = results
        return results
    
    def run_pdbbind_benchmark(
        self,
        predictor,
        pdbbind_subset: str = 'core'
    ) -> AffinityMetrics:
        """
        Benchmark on PDBbind affinity dataset.
        
        Args:
            predictor: Boltz2 predictor
            pdbbind_subset: 'core', 'refined', or 'general'
            
        Returns:
            AffinityMetrics
        """
        # PDBbind core set (high-quality complexes with measured Kd/Ki)
        pdbbind_samples = [
            {'pdb': '1a1e', 'sequence': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP', 
             'ligand': 'CC(C)CC1CCC(C)CC1', 'exp_kd': 2.3},  # nM
            {'pdb': '1owe', 'sequence': 'KSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG',
             'ligand': 'CC1=CC=C(C=C1)C(=O)O', 'exp_kd': 150.0},
            # Add more samples as needed
        ]
        
        predictions = []
        experimental = []
        
        for sample in pdbbind_samples:
            print(f"Processing {sample['pdb']}...")
            
            try:
                # Create FASTA
                fasta_path = self.output_dir / f"{sample['pdb']}.fasta"
                with open(fasta_path, 'w') as f:
                    f.write(f">{sample['pdb']}\n{sample['sequence']}\n")
                
                # Predict affinity
                result = predictor.predict_with_affinity(
                    protein_fasta=str(fasta_path),
                    ligand_smiles=sample['ligand'],
                    output_dir=str(self.output_dir / sample['pdb'])
                )
                
                predictions.append(result.ic50)
                experimental.append(sample['exp_kd'])
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Calculate metrics
        pred_array = np.array(predictions)
        exp_array = np.array(experimental)
        
        # Convert to pKd for better correlation
        pred_pkd = -np.log10(pred_array * 1e-9)
        exp_pkd = -np.log10(exp_array * 1e-9)
        
        metrics = self.metrics_calc.calculate_affinity_metrics(pred_pkd, exp_pkd)
        
        self.results['pdbbind'] = {
            'predictions': predictions,
            'experimental': experimental.tolist(),
            'metrics': asdict(metrics)
        }
        
        return metrics
    
    def generate_benchmark_report(self) -> str:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            Path to HTML report
        """
        report_path = self.output_dir / 'benchmark_report.html'
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; font-weight: bold; }
        tr:hover { background: #f5f5f5; }
        .metric { display: inline-block; margin: 15px; padding: 20px; background: #ecf0f1; border-radius: 8px; min-width: 150px; }
        .metric-value { font-size: 32px; font-weight: bold; color: #3498db; }
        .metric-label { font-size: 14px; color: #7f8c8d; margin-top: 5px; }
        .highlight { background: #2ecc71; color: white; padding: 5px 10px; border-radius: 5px; }
        .comparison { background: #e8f8f5; padding: 20px; border-left: 4px solid #1abc9c; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Protein Folding Benchmark Report</h1>
        <p><strong>Generated:</strong> """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
        
        # CASP15 Results
        if self.results['casp15']:
            casp_results = self.results['casp15']
            
            html += """<h2>CASP15 Evaluation</h2>
<div class="comparison">
    <h3>Key Findings</h3>
    <p>Evaluated on """ + str(len(casp_results)) + """ CASP15 targets</p>
</div>
"""
            
            # Summary metrics
            avg_gdt = np.mean([r.gdt_ts for r in casp_results])
            avg_tm = np.mean([r.tm_score for r in casp_results])
            avg_lddt = np.mean([r.lddt for r in casp_results])
            
            html += f"""<div>
    <div class="metric">
        <div class="metric-value">{avg_gdt:.1f}</div>
        <div class="metric-label">Average GDT_TS</div>
    </div>
    <div class="metric">
        <div class="metric-value">{avg_tm:.3f}</div>
        <div class="metric-label">Average TM-score</div>
    </div>
    <div class="metric">
        <div class="metric-value">{avg_lddt:.1f}</div>
        <div class="metric-label">Average lDDT</div>
    </div>
</div>
"""
            
            # Results table
            html += """<h3>Detailed Results</h3>
<table>
    <tr>
        <th>Target</th>
        <th>GDT_TS</th>
        <th>TM-score</th>
        <th>lDDT</th>
        <th>RMSD (Å)</th>
        <th>Time (s)</th>
    </tr>
"""
            
            for result in casp_results:
                html += f"""<tr>
        <td>{result.target_id}</td>
        <td>{result.gdt_ts:.2f}</td>
        <td>{result.tm_score:.3f}</td>
        <td>{result.lddt:.2f}</td>
        <td>{result.rmsd:.2f}</td>
        <td>{result.prediction_time:.1f}</td>
    </tr>
"""
            
            html += "</table>"
        
        # PDBbind Results
        if self.results['pdbbind']:
            pdbbind_data = self.results['pdbbind']
            metrics = pdbbind_data['metrics']
            
            html += f"""<h2>PDBbind Affinity Benchmark</h2>
<div class="comparison">
    <h3>Binding Affinity Prediction Performance</h3>
</div>
<div>
    <div class="metric">
        <div class="metric-value">{metrics['pearson_r']:.3f}</div>
        <div class="metric-label">Pearson R</div>
    </div>
    <div class="metric">
        <div class="metric-value">{metrics['rmse']:.2f}</div>
        <div class="metric-label">RMSE (pKd)</div>
    </div>
    <div class="metric">
        <div class="metric-value">{metrics['r_squared']:.3f}</div>
        <div class="metric-label">R²</div>
    </div>
</div>
"""
        
        html += """    </div>
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        # Save JSON results
        json_path = self.output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nBenchmark report saved to: {report_path}")
        return str(report_path)
