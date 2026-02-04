# Usage Guide

## Quick Start

### Command Line Interface

#### AlphaFold 3 Prediction

```bash
protein-predict \
  --fasta data/examples/protein.fasta \
  --output output/af3_results \
  --visualize
```

#### Boltz-2 Affinity Prediction

```bash
protein-affinity \
  --fasta data/examples/protein.fasta \
  --ligand 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --output output/boltz2_results \
  --affinity \
  --visualize
```

### Python API

#### Basic AlphaFold 3 Usage

```python
from src.alphafold3 import AlphaFold3Predictor
from src.visualization import StructureViewer

# Initialize
predictor = AlphaFold3Predictor(model_dir='./models/alphafold3')

# Predict
result = predictor.predict(
    fasta_path='protein.fasta',
    output_dir='output',
    num_recycling=3
)

# Visualize
viewer = StructureViewer()
view = viewer.show_structure(
    result.pdb_path,
    confidence=result.plddt
)
```

#### Basic Boltz-2 Usage

```python
from src.boltz2 import Boltz2Predictor, AffinityCalculator

# Initialize
predictor = Boltz2Predictor(model_dir='./models/boltz2')

# Predict with affinity
result = predictor.predict_with_affinity(
    protein_fasta='protein.fasta',
    ligand_smiles='CC(=O)O',
    output_dir='output'
)

# Analyze affinity
print(f"IC50: {result.ic50} nM")
print(f"ΔG: {result.delta_g} kcal/mol")
print(f"Class: {AffinityCalculator.classify_affinity(result.ic50)}")
```

## Advanced Usage

### Protein Complex Prediction

```python
from src.alphafold3 import AlphaFold3Predictor

predictor = AlphaFold3Predictor()

# Protein-DNA complex
result = predictor.predict_complex(
    protein_fasta='protein.fasta',
    dna_fasta='dna.fasta',
    output_dir='complex_output'
)
```

### Multiple Ligand Screening

```python
from src.boltz2 import Boltz2Predictor

predictor = Boltz2Predictor()
ligands = ['CCO', 'CC(=O)O', 'CC(C)O']  # SMILES strings

results = []
for ligand in ligands:
    result = predictor.predict_with_affinity(
        protein_fasta='protein.fasta',
        ligand_smiles=ligand,
        output_dir=f'output_{ligand}'
    )
    results.append(result)

# Find best binder
best = min(results, key=lambda r: r.ic50)
print(f"Best IC50: {best.ic50} nM")
```

### Complete Pipeline

```python
from src.pipeline import ProteinPipeline

pipeline = ProteinPipeline(
    alphafold3_weights='./models/alphafold3',
    boltz2_weights='./models/boltz2'
)

results = pipeline.run(
    sequence='MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    ligands=['CCO', 'CC(=O)O'],
    generate_report=True,
    experiment_name='my_experiment'
)

print(f"Report: {results.report_path}")
```

## Visualization

### Confidence Plots

```python
from src.visualization import ConfidencePlotter
import numpy as np

plotter = ConfidencePlotter()

# pLDDT plot
plotter.plot_plddt(
    plddt=result.plddt,
    title="Confidence Scores",
    save_path="plddt.png"
)

# PAE heatmap
plotter.plot_pae(
    pae=result.pae,
    title="Predicted Aligned Error",
    save_path="pae.png"
)

# Combined summary
plotter.plot_confidence_summary(
    plddt=result.plddt,
    pae=result.pae,
    ptm=result.ptm,
    save_path="summary.png"
)
```

### Interactive 3D Views

```python
from src.visualization import StructureViewer

viewer = StructureViewer(width=1000, height=800)

# Simple view
view = viewer.show_structure(
    'structure.pdb',
    style='cartoon'
)

# Complex view
view = viewer.show_complex(
    'complex.pdb',
    protein_chain='A',
    ligand_chain='B'
)

# Comparison view
view = viewer.create_comparison_view(
    pdb_paths=['model1.pdb', 'model2.pdb'],
    labels=['AlphaFold 3', 'Boltz-2']
)
```

## Configuration

### AlphaFold 3 Options

```python
result = predictor.predict(
    fasta_path='input.fasta',
    output_dir='output',
    num_recycling=3,           # More iterations = better quality
    num_diffusion_steps=200,   # Diffusion denoising steps
    use_templates=True,        # Use PDB templates
    msa_mode='full',           # 'full', 'reduced', or 'none'
    random_seed=42             # For reproducibility
)
```

### Boltz-2 Options

```python
result = predictor.predict(
    fasta_path='input.fasta',
    output_dir='output',
    num_recycling=3,           # Recycling iterations
    use_msa=True,              # Fetch MSA data
    random_seed=42             # For reproducibility
)
```

## Best Practices

### Sequence Preparation

1. **Remove signal peptides**: Use SignalP for prediction
2. **Check for disorder**: Use IUPred or similar tools
3. **Validate sequence**: Ensure standard amino acids only

### Performance Tips

1. **Use GPU when available**: 10-100x speedup
2. **Batch predictions**: Process multiple sequences together
3. **Cache MSAs**: Reuse MSAs for similar sequences
4. **Reduce recycling**: Use 1-2 iterations for speed

### Result Interpretation

#### pLDDT Scores
- **>90**: Very high confidence
- **70-90**: High confidence
- **50-70**: Low confidence
- **<50**: Very low confidence (likely disordered)

#### Binding Affinity
- **IC50 <1 nM**: Very strong binder
- **1-10 nM**: Strong binder
- **10-100 nM**: Moderate binder
- **100-1000 nM**: Weak binder
- **>1000 nM**: Very weak/non-binder

## Examples

See `notebooks/` directory for complete examples:
- `alphafold3_demo.ipynb`: AlphaFold 3 basics
- `boltz2_demo.ipynb`: Boltz-2 affinity prediction
- `combined_analysis.ipynb`: Complete pipeline
