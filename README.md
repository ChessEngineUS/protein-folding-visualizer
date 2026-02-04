# Protein Folding Visualizer: AlphaFold 3 & Boltz-2

State-of-the-art protein structure prediction and visualization using AlphaFold 3 and Boltz-2 with interactive 3D visualization, binding affinity prediction, and comprehensive quality metrics.

## Features

- **AlphaFold 3 Integration**: Diffusion-based biomolecular structure prediction with 50-100% improvement over previous methods
- **Boltz-2 Binding Affinity**: Fast, accurate binding affinity prediction (1000x faster than FEP methods)
- **Interactive 3D Visualization**: Real-time structure viewing with py3Dmol and NGLView
- **Comprehensive Metrics**: pLDDT confidence scores, PAE matrices, binding probabilities
- **Multi-Chain Complex Support**: Protein-protein, protein-ligand, protein-DNA/RNA interactions
- **Automated Pipeline**: End-to-end workflow from FASTA to publication-quality visualizations

## Installation

### Requirements
- Python 3.11+
- CUDA 12.1+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended)
- 100GB+ disk space for model weights

### Quick Start

```bash
# Clone repository
git clone https://github.com/ChessEngineUS/protein-folding-visualizer.git
cd protein-folding-visualizer

# Create conda environment
conda env create -f environment.yml
conda activate protein-viz

# Install dependencies
pip install -e .

# Download model weights
python scripts/download_models.py --alphafold3 --boltz2
```

## Usage

### AlphaFold 3 Prediction

```python
from src.alphafold3 import AlphaFold3Predictor
from src.visualization import StructureViewer

# Initialize predictor
predictor = AlphaFold3Predictor(model_dir='./models/alphafold3')

# Run prediction
result = predictor.predict(
    fasta_path='data/examples/protein.fasta',
    output_dir='data/outputs',
    num_recycling=3,
    use_templates=True
)

# Visualize
viewer = StructureViewer()
viewer.show_structure(result.pdb_path, confidence=result.plddt)
```

### Boltz-2 Affinity Prediction

```python
from src.boltz2 import Boltz2Predictor, AffinityCalculator

# Initialize predictor
predictor = Boltz2Predictor(model_dir='./models/boltz2')

# Predict structure + affinity
result = predictor.predict_with_affinity(
    protein_fasta='protein.fasta',
    ligand_smiles='CCO',  # Ethanol example
    output_dir='data/outputs'
)

# Extract affinity metrics
print(f"IC50: {result.ic50} nM")
print(f"ΔG: {result.delta_g} kcal/mol")
print(f"Binding Probability: {result.binding_probability:.2%}")
```

### Combined Pipeline

```python
from src.pipeline import ProteinPipeline

# Initialize pipeline with both models
pipeline = ProteinPipeline(
    alphafold3_weights='./models/alphafold3',
    boltz2_weights='./models/boltz2'
)

# Run comprehensive analysis
results = pipeline.run(
    sequence='MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    ligands=['CCO', 'CC(=O)O'],  # Multiple ligands
    generate_report=True
)

# Results include:
# - AF3 structure predictions
# - Boltz-2 binding poses
# - Affinity predictions
# - Interactive HTML report
```

## Architecture

### AlphaFold 3
- Diffusion-based structure generation
- Enhanced accuracy for protein-ligand, protein-DNA/RNA complexes
- Confidence metrics: pLDDT, PAE, pTM, ipTM

### Boltz-2
- Rapid binding affinity prediction
- IC50, ΔG, and binding probability estimates
- 1000x speedup over traditional FEP methods

### Visualization
- Interactive 3D viewers (py3Dmol, NGLView)
- Confidence-based coloring schemes
- PAE matrix heatmaps
- Multi-structure comparison views

## Project Structure

```
protein-folding-visualizer/
├── src/
│   ├── alphafold3/          # AlphaFold 3 prediction
│   ├── boltz2/              # Boltz-2 prediction and affinity
│   ├── visualization/       # 3D visualization tools
│   └── pipeline/            # Orchestration pipeline
├── notebooks/               # Jupyter examples
├── scripts/                 # CLI tools
├── tests/                   # Unit tests
├── data/
│   ├── examples/           # Example inputs
│   └── outputs/            # Prediction outputs
└── docs/                   # Documentation
```

## Key Citations

- **AlphaFold 3**: Abramson et al., Nature (2024). DOI: 10.1038/s41586-024-07487-w
- **Boltz-2**: Wohlwend et al., bioRxiv (2025). DOI: 10.1101/2025.06.14.659707
- **Original AlphaFold**: Jumper et al., Nature (2021). DOI: 10.1038/s41586-021-03819-2

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built on the foundational work of:
- Google DeepMind (AlphaFold 3)
- MIT & Recursion (Boltz-2)
- Open-source structural biology community

## Support

For questions or issues:
- Open a GitHub issue
- Check documentation in `docs/`
- Review example notebooks in `notebooks/`
