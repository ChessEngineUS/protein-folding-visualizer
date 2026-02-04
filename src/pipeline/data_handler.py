"""Data handling utilities for pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil


class DataHandler:
    """Manage input/output data for pipeline."""
    
    def __init__(self, base_dir: str = './data'):
        self.base_dir = Path(base_dir)
        self.examples_dir = self.base_dir / 'examples'
        self.outputs_dir = self.base_dir / 'outputs'
        
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment(
        self,
        experiment_name: str,
        results: Dict,
        files: Dict[str, str]
    ):
        """
        Save experiment results.
        
        Args:
            experiment_name: Name of experiment
            results: Dictionary of results
            files: Dictionary mapping file types to paths
        """
        exp_dir = self.outputs_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_path = exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Copy files
        for file_type, file_path in files.items():
            if Path(file_path).exists():
                dest = exp_dir / f"{file_type}{Path(file_path).suffix}"
                shutil.copy2(file_path, dest)
    
    def load_experiment(self, experiment_name: str) -> Dict:
        """
        Load experiment results.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Dictionary of results
        """
        exp_dir = self.outputs_dir / experiment_name
        results_path = exp_dir / 'results.json'
        
        if not results_path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_name}")
        
        with open(results_path) as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """
        List all saved experiments.
        
        Returns:
            List of experiment names
        """
        return [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]
    
    def cleanup_experiment(self, experiment_name: str):
        """
        Delete experiment data.
        
        Args:
            experiment_name: Name of experiment to delete
        """
        exp_dir = self.outputs_dir / experiment_name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
    
    def get_example_sequences(self) -> Dict[str, str]:
        """
        Get example protein sequences.
        
        Returns:
            Dictionary mapping names to sequences
        """
        examples = {
            'insulin': 'MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN',
            'lysozyme': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
            'ubiquitin': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
        }
        return examples
