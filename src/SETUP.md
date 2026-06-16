# Environment Setup Guide

## Option 1: Using Conda (Recommended)

1. Create a new conda environment:
```bash
conda create -n push_physics python=3.10
conda activate push_physics
```

2. Install PyTorch 2.4.0 with CUDA support:
```bash
# For CUDA 12.1
conda install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only
conda install pytorch=2.4.0 torchvision torchaudio cpuonly -c pytorch
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Option 2: Using pip venv

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install PyTorch 2.4.0:
```bash
# For CUDA 12.1
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Install other dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify Installation

Run the test script to verify your setup:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.4.0
CUDA available: True  # If CUDA is installed
```

## Development Setup (Optional)

1. Install pre-commit hooks for code quality:
```bash
pip install pre-commit
pre-commit install
```

2. Setup JupyterLab for development:
```bash
jupyter lab
```

## Common Issues

### CUDA Issues
- Ensure your NVIDIA drivers are up to date
- Required CUDA version: 12.1
- Use `nvidia-smi` to verify CUDA installation

### Package Conflicts
- If you encounter package conflicts, try:
```bash
pip uninstall torch torchvision torchaudio  # Remove existing PyTorch
pip install -r requirements.txt  # Reinstall dependencies
```

### Memory Issues
- For large datasets, you might need to adjust:
  - Batch size in config/default.yaml
  - Number of workers in DataLoader
  - CUDA memory settings


## System Requirements

- Python 3.10 or higher
- CUDA 11.8-12.2 (for GPU support)
- NVIDIA GPU with 4GB+ memory (recommended)
- 16GB RAM (minimum)
- 30GB free disk space