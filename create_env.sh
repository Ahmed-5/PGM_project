# Create conda environment
conda create -n relaxed-equiv python=3.10
conda activate relaxed-equiv

# Install PyTorch
pip install torch torchvision

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-geometric

# Install e3nn (for future equivariance work)
pip install e3nn

# Clone reference repositories
git clone https://github.com/atomicarchitects/RelaxedE3NN
git clone https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets

# Install reference repositories