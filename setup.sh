#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up environment..."

# clear previous venv if it exists (optional, but good for clean slate if re-running)
rm -rf venv 

# Create a virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
REQUIREMENTS_FILE="requirements.txt"
pip install -r "$REQUIREMENTS_FILE"



echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
