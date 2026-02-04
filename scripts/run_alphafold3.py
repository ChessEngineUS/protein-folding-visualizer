#!/usr/bin/env python
"""Command-line interface for AlphaFold 3 predictions."""

import click
from pathlib import Path
from src.alphafold3 import AlphaFold3Predictor
from src.visualization import StructureViewer, ConfidencePlotter


@click.command()
@click.option('--fasta', '-f', required=True, help='Input FASTA file')
@click.option('--output', '-o', default='./output', help='Output directory')
@click.option('--model-dir', '-m', default='./models/alphafold3', help='Model directory')
@click.option('--recycling', '-r', default=3, help='Number of recycling iterations')
@click.option('--diffusion-steps', '-d', default=200, help='Diffusion denoising steps')
@click.option('--no-templates', is_flag=True, help='Disable template search')
@click.option('--msa-mode', default='full', type=click.Choice(['full', 'reduced', 'none']))
@click.option('--seed', type=int, help='Random seed')
@click.option('--visualize', is_flag=True, help='Generate visualizations')
def main(fasta, output, model_dir, recycling, diffusion_steps, no_templates, msa_mode, seed, visualize):
    """Run AlphaFold 3 structure prediction."""
    
    click.echo(f"AlphaFold 3 Prediction")
    click.echo(f"Input: {fasta}")
    click.echo(f"Output: {output}")
    
    # Initialize predictor
    predictor = AlphaFold3Predictor(model_dir=model_dir)
    
    # Run prediction
    with click.progressbar(length=1, label='Predicting structure') as bar:
        result = predictor.predict(
            fasta_path=fasta,
            output_dir=output,
            num_recycling=recycling,
            num_diffusion_steps=diffusion_steps,
            use_templates=not no_templates,
            msa_mode=msa_mode,
            random_seed=seed
        )
        bar.update(1)
    
    # Print results
    click.echo(f"\n✓ Prediction complete!")
    click.echo(f"  PDB file: {result.pdb_path}")
    click.echo(f"  Mean pLDDT: {result.plddt.mean():.2f}")
    click.echo(f"  pTM: {result.ptm:.3f}")
    if result.iptm:
        click.echo(f"  ipTM: {result.iptm:.3f}")
    
    # Generate visualizations
    if visualize:
        click.echo("\nGenerating visualizations...")
        plotter = ConfidencePlotter()
        
        output_path = Path(output)
        plotter.plot_plddt(
            result.plddt,
            save_path=str(output_path / 'plddt.png')
        )
        plotter.plot_pae(
            result.pae,
            save_path=str(output_path / 'pae.png')
        )
        
        click.echo("  ✓ Plots saved")


if __name__ == '__main__':
    main()
