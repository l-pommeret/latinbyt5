#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --output=install.out
#SBATCH --error=install.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Ensure we are in the right place
cd ~/propositions_atomiques/pommeret/latinbyt5

echo "Installing uv..."
# Install uv locally (doesn't need root, works on old glibc)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add local bin to path (default install location)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "Creating virtual environment with Python 3.12..."
# uv automatically downloads a standalone python if needed
uv venv .venv --python 3.12
source .venv/bin/activate

echo "Installing dependencies..."
# uv pip install is extremely fast and resolves wheels better than pip
uv pip install -r requirements.txt

# Manually ensure torch is the cuda version if needed, 
# though 'uv pip install torch' usually grabs the right wheel. 
# For explicit CUDA 12:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installation complete!"
echo "To activate: source .venv/bin/activate"
