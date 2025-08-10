#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 0: Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "=== Starting Malicious URL Detection Pipeline ==="

# Step 1: Merge datasets
echo "[1/4] Merging datasets..."
python3 merge_datasets.py

# Step 2: Extract features
echo "[2/4] Extracting features..."
python3 FeatureExtractor.py

# Step 3: Train ML models
echo "[3/4] Training Machine Learning models..."
python3 TrainAllModels_ML.py

# Step 4: Train Deep Learning models
echo "[4/4] Training Deep Learning models..."
python3 TrainAllModels_DL.py

echo "=== Pipeline completed successfully! Results are in the 'results/' folder. ==="
