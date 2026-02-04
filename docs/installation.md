# Installation Guide

## System Requirements

### Hardware
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
  - CUDA 12.1 or higher
  - Compute capability 7.0+
- **Storage**: 100GB+ free space for model weights and databases

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.11 or higher
- **Conda**: Recommended for environment management

## Quick Installation

### 1. Clone Repository

```bash
git clone https://github.com/ChessEngineUS/protein-folding-visualizer.git
cd protein-folding-visualizer
```

### 2. Create Environment

#### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate protein-viz
```

#### Using pip + venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Package

```bash
pip install -e .
```

### 4. Download Model Weights

```bash
python scripts/download_models.py --alphafold3 --boltz2
```

This will download:
- AlphaFold 3 weights (~50GB)
- Boltz-2 weights (~5GB)

**Note**: Downloads may take 30-60 minutes depending on connection speed.

## Detailed Installation

### AlphaFold 3 Setup

1. **Download Official Implementation**

```bash
git clone https://github.com/google-deepmind/alphafold3.git
cd alphafold3
pip install -r requirements.txt
```

2. **Download Databases** (Optional but recommended)

```bash
# UniRef90 (~60GB)
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz

# PDB templates (~200GB)
wget https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/
```

3. **Configure Paths**

Edit `config/alphafold3_config.yaml`:

```yaml
model_dir: ./models/alphafold3
database_dir: ./databases/alphafold
use_gpu: true
num_gpus: 1
```

### Boltz-2 Setup

1. **Install Boltz Package**

```bash
pip install boltz
```

2. **Verify Installation**

```bash
boltz --version
```

3. **Configure Paths**

Edit `config/boltz2_config.yaml`:

```yaml
model_dir: ./models/boltz2
use_gpu: true
device: cuda:0
```

## GPU Setup

### CUDA Installation

```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1
```

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Common Issues

#### "CUDA out of memory"

**Solution**: Reduce batch size or use CPU mode:

```python
predictor = AlphaFold3Predictor(use_gpu=False)
```

#### "Model weights not found"

**Solution**: Verify download completed:

```bash
ls -lh models/alphafold3/
ls -lh models/boltz2/
```

#### "Import error: py3Dmol"

**Solution**: Install visualization dependencies:

```bash
pip install py3Dmol nglview jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py nglview
```

### Performance Optimization

#### For CPU-only systems:

```python
import torch
torch.set_num_threads(8)  # Set to your CPU core count
```

#### For multi-GPU systems:

```python
predictor = AlphaFold3Predictor(
    use_gpu=True,
    num_gpus=2  # Use 2 GPUs
)
```

## Testing Installation

Run test suite:

```bash
pytest tests/
```

Run quick validation:

```bash
python -c "from src.alphafold3 import AlphaFold3Predictor; print('AlphaFold 3 OK')"
python -c "from src.boltz2 import Boltz2Predictor; print('Boltz-2 OK')"
```

## Next Steps

- Review [Usage Guide](usage.md)
- Try example notebooks in `notebooks/`
- Check [API Reference](api_reference.md)
