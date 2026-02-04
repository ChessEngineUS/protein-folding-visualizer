#!/usr/bin/env python
"""Command-line interface for visualizing prediction results."""

import click
from pathlib import Path
import json
import numpy as np
from src.visualization import StructureViewer, ConfidencePlotter


@click.command()
@click.option('--pdb', '-p', required=True, help='PDB structure file')
@click.option('--confidence', '-c', help='Confidence JSON file')
@click.option('--output', '-o', default='./visualizations', help='Output directory')
@click.option('--style', default='cartoon', type=click.Choice(['cartoon', 'stick', 'sphere']))
@click.option('--interactive', is_flag=True, help='Generate interactive HTML view')
def main(pdb, confidence, output, style, interactive):
    """Visualize protein structure predictions."""
    
    click.echo(f"Visualizing Structure")
    click.echo(f"PDB: {pdb}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load confidence data if provided
    plddt = None
    pae = None
    if confidence:
        with open(confidence) as f:
            conf_data = json.load(f)
            plddt = np.array(conf_data.get('plddt', []))
            pae = np.array(conf_data.get('pae', []))
    
    # Generate static plots
    if plddt is not None and len(plddt) > 0:
        click.echo("Generating confidence plots...")
        plotter = ConfidencePlotter()
        
        plotter.plot_plddt(
            plddt,
            save_path=str(output_path / 'plddt.png')
        )
        click.echo(f"  ✓ pLDDT plot saved")
        
        if pae is not None and len(pae) > 0:
            plotter.plot_pae(
                pae,
                save_path=str(output_path / 'pae.png')
            )
            click.echo(f"  ✓ PAE plot saved")
    
    # Generate interactive view
    if interactive:
        click.echo("Generating interactive 3D view...")
        viewer = StructureViewer()
        view = viewer.show_structure(
            pdb,
            confidence=plddt,
            style=style,
            color_by_confidence=(plddt is not None)
        )
        
        # Save as HTML
        html_path = output_path / 'structure_3d.html'
        with open(html_path, 'w') as f:
            f.write(view._make_html())
        
        click.echo(f"  ✓ Interactive view saved to {html_path}")
    
    click.echo("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
