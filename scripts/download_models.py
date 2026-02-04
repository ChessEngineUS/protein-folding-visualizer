#!/usr/bin/env python
"""Download model weights for AlphaFold 3 and Boltz-2."""

import click
import urllib.request
from pathlib import Path
from tqdm import tqdm
import tarfile
import zipfile


MODEL_URLS = {
    'alphafold3': 'https://storage.googleapis.com/deepmind-alphafold3/alphafold3_weights.tar.gz',
    'boltz2': 'https://github.com/jwohlwend/boltz/releases/download/v2.0/boltz2_weights.zip'
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


@click.command()
@click.option('--alphafold3', is_flag=True, help='Download AlphaFold 3 weights')
@click.option('--boltz2', is_flag=True, help='Download Boltz-2 weights')
@click.option('--output-dir', '-o', default='./models', help='Output directory for models')
def main(alphafold3, boltz2, output_dir):
    """Download model weights."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not alphafold3 and not boltz2:
        click.echo("Please specify at least one model to download (--alphafold3 or --boltz2)")
        return
    
    if alphafold3:
        click.echo("Downloading AlphaFold 3 weights...")
        af3_dir = output_path / 'alphafold3'
        af3_dir.mkdir(exist_ok=True)
        
        archive_path = af3_dir / 'weights.tar.gz'
        download_url(MODEL_URLS['alphafold3'], str(archive_path))
        
        click.echo("Extracting...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(af3_dir)
        
        archive_path.unlink()
        click.echo("✓ AlphaFold 3 weights downloaded")
    
    if boltz2:
        click.echo("\nDownloading Boltz-2 weights...")
        boltz_dir = output_path / 'boltz2'
        boltz_dir.mkdir(exist_ok=True)
        
        archive_path = boltz_dir / 'weights.zip'
        download_url(MODEL_URLS['boltz2'], str(archive_path))
        
        click.echo("Extracting...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(boltz_dir)
        
        archive_path.unlink()
        click.echo("✓ Boltz-2 weights downloaded")
    
    click.echo("\n✓ All downloads complete!")


if __name__ == '__main__':
    main()
