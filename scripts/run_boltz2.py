#!/usr/bin/env python
"""Command-line interface for Boltz-2 predictions."""

import click
from pathlib import Path
from src.boltz2 import Boltz2Predictor, AffinityCalculator
from src.visualization import StructureViewer, ConfidencePlotter


@click.command()
@click.option('--fasta', '-f', required=True, help='Protein FASTA file')
@click.option('--ligand', '-l', help='Ligand SMILES string')
@click.option('--output', '-o', default='./output', help='Output directory')
@click.option('--model-dir', '-m', default='./models/boltz2', help='Model directory')
@click.option('--recycling', '-r', default=3, help='Number of recycling steps')
@click.option('--no-msa', is_flag=True, help='Disable MSA generation')
@click.option('--seed', type=int, help='Random seed')
@click.option('--affinity', is_flag=True, help='Calculate binding affinity')
@click.option('--visualize', is_flag=True, help='Generate visualizations')
def main(fasta, ligand, output, model_dir, recycling, no_msa, seed, affinity, visualize):
    """Run Boltz-2 structure and affinity prediction."""
    
    click.echo(f"Boltz-2 Prediction")
    click.echo(f"Protein: {fasta}")
    if ligand:
        click.echo(f"Ligand: {ligand}")
    click.echo(f"Output: {output}")
    
    # Initialize predictor
    predictor = Boltz2Predictor(model_dir=model_dir)
    
    # Run prediction
    with click.progressbar(length=1, label='Predicting structure') as bar:
        if affinity and ligand:
            result = predictor.predict_with_affinity(
                protein_fasta=fasta,
                ligand_smiles=ligand,
                output_dir=output
            )
        else:
            result = predictor.predict(
                fasta_path=fasta,
                output_dir=output,
                num_recycling=recycling,
                use_msa=not no_msa,
                random_seed=seed
            )
        bar.update(1)
    
    # Print results
    click.echo(f"\n✓ Prediction complete!")
    click.echo(f"  PDB file: {result.pdb_path}")
    click.echo(f"  Mean pLDDT: {result.plddt.mean():.2f}")
    
    if result.ic50 is not None:
        click.echo(f"\n  Binding Affinity:")
        click.echo(f"    IC50: {result.ic50:.2f} nM")
        click.echo(f"    ΔG: {result.delta_g:.2f} kcal/mol")
        click.echo(f"    Binding Probability: {result.binding_probability:.2%}")
        click.echo(f"    Classification: {AffinityCalculator.classify_affinity(result.ic50)}")
    
    # Generate visualizations
    if visualize:
        click.echo("\nGenerating visualizations...")
        plotter = ConfidencePlotter()
        
        output_path = Path(output)
        plotter.plot_plddt(
            result.plddt,
            save_path=str(output_path / 'plddt.png')
        )
        
        click.echo("  ✓ Plots saved")


if __name__ == '__main__':
    main()
