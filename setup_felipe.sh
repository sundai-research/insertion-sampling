#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Download and install Miniconda
echo "Downloading Miniconda installer..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

echo "Installing Miniconda..."
bash miniconda.sh -b

# Initialize conda in this shell session
echo "Initializing conda..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. Clone the repositories
echo "Cloning repositories..."
git clone https://github.com/GAIR-NLP/LIMO.git
git clone https://github.com/agentica-project/deepscaler.git

# 3. Create a conda environment called "felipe_test" with Python
echo "Creating conda environment 'felipe_test'..."
conda create -n felipe_test python -y

# 4. Activate the environment
echo "Activating conda environment 'felipe_test'..."
conda activate felipe_test

# 5. Change directory into the insertion-sampling repository
echo "Changing directory to insertion-sampling..."
cd insertion-sampling

# 6. Install the required Python packages
echo "Installing required Python packages..."
pip install numpy vllm transformers

echo "Setup complete!"
