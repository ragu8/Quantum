#!/bin/bash

# Update package list and install Python and venv if not already installed
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

jupyter

# Create a virtual environment
sudo python3 -m venv quantum

# Activate the virtual environment
source quantum/bin/activate

# Create a requirements.txt file
cat <<EOT >> requirements.txt
qiskit
qiskit-machine-learning
scikit-learn
seaborn
matplotlib
yfinance
jupyter
EOT

# Install the packages
pip install -r requirements.txt

echo "Setup complete. Virtual environment 'quantum' is ready to use."
echo "Start 'jupyter notebook' "


