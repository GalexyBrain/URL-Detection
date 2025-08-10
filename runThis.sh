#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 0: Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "=== Starting Malicious URL Detection Pipeline ==="

# Step 1: Merge datasets
echo "[1/5] Merging datasets..."
python3 merge_datasets.py

# Step 2: Extract features
echo "[2/5] Extracting features..."
python3 FeatureExtractor.py

# Step 3: Train ML models
echo "[3/5] Training Machine Learning models..."
python3 TrainAllModels_ML.py

# Step 4: Train Deep Learning models
echo "[4/5] Training Deep Learning models..."
python3 TrainAllModels_DL.py

# Step 5: Train Deep Learning models
echo "[5/5] Attacking all models..."
python3 attackModels.py

echo "=== Pipeline completed successfully! Results are in the 'results/' folder. ==="
