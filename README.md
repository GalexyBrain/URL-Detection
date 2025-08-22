
# Malicious URL Detection Framework

A comprehensive machine learning and deep learning framework for detecting malicious URLs with adversarial robustness testing and defense mechanisms.

## 🚀 Project Overview

This framework implements a complete pipeline for malicious URL detection, including:
- **Feature extraction** from URLs (lexical and host-based features)
- **Model training** using both traditional ML and deep learning approaches
- **Adversarial evaluation** to test model robustness
- **Defense mechanisms** to protect against adversarial attacks
- **Comprehensive evaluation** with detailed metrics and visualizations

## 📁 Project Structure

### Core Scripts by Category

#### 🔧 **Data Preparation**
| Script | Purpose | Output |
|--------|---------|--------|
| `merge_datasets.py` | Combines multiple URL datasets | `merged_url_dataset.csv` |
| `FeatureExtractor.py` | Extracts features from URLs | `features_extracted.csv` + distribution plots |

#### 🎯 **Model Training**
| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `TrainAllModels_ML.py` | Trains traditional ML models | `models/` + `results/` |
| `TrainAllModels_DL.py` | Trains deep learning models | `models/` + `results/` |
| `TrainAllFeatureDefensiveModels.py` | Trains 3-class models (benign/malicious/adversarial) | `models_base3/` + `results_base3/` |

#### ⚔️ **Adversarial Evaluation**
| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `attackModels.py` | Tests models against adversarial attacks | `results_evaluation/` |

#### 🛡️ **Defense Systems**
| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `train_DefensiveLightGBM.py` | Trains adversarial detector for features | `models/Defense-LGBM/` + `results_defense_features/Defense-LGBM/` |
| `train_DefensiveCharCNN.py` | Trains adversarial detector for character models | `models/Defense-CharCNN/` |
| `defence_FeatureModels.py` | Evaluates defense on feature models | `results_defense_feat_stream/` |
| `defence_FeatureModels3Class.py` | 3-class defense evaluation | `results_defence_features_3class/` |
| `defense_CharacterModels.py` | Evaluates defense on character models | `results_defense_char_stream/` |

#### 🔬 **Defense Data Preparation**
| Script | Purpose | Output |
|--------|---------|--------|
| `prep_defensive_dataset_features.py` | Creates adversarial training data for features | `features_adversarial_defense_dataset.csv` |
| `prep_defensive_dataset_characters.py` | Creates character-based defense data | Character defense dataset |
| `prep_defence_dataset_features_for_training.py` | Prepares features for defense training | Defense training dataset |
| `remove_dups_defensive.py` | Removes duplicates from defense datasets | Cleaned defense datasets |

#### 🧪 **Utility Scripts**
| Script | Purpose | Output |
|--------|---------|--------|
| `defense.py` | Defense utility functions | N/A (utility) |
| `runThis.sh` | Runs complete pipeline | Executes all phases |

---

## 📊 Detailed Script Documentation

### **Data Preparation Scripts**

#### `merge_datasets.py`
- **Purpose**: Combines benign and malicious URL datasets
- **Inputs**: `url_dataset_1.csv`, `url_dataset_2.csv`
- **Outputs**: `merged_url_dataset.csv`
- **Contains**: Combined URL dataset with labels

#### `FeatureExtractor.py`
- **Purpose**: Extracts numerical features from raw URLs
- **Inputs**: `merged_url_dataset.csv`
- **Outputs**: 
  - `features_extracted.csv` - Main feature dataset
  - `all_feature_distributions.png` - Feature distribution visualizations
  - `url_length_distribution_benign.png` - Benign URL length distribution
  - `url_length_distribution_malicious.png` - Malicious URL length distribution
  - `url_length_distribution_compare.png` - Comparison of URL lengths
  - `url_length_cdf_compare.png` - CDF comparison of URL lengths
- **Contains**: 
  - Original URLs
  - 30+ extracted features (URL length, special characters, domain info, etc.)
  - Binary labels (0=benign, 1=malicious)

### **Model Training Scripts**

#### `TrainAllModels_ML.py`
- **Purpose**: Trains traditional machine learning models
- **Inputs**: `features_extracted.csv`
- **Models Trained**: Logistic Regression, Calibrated LinearSVC, Naive Bayes, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM
- **Outputs**:
  - **Models**: `models/{ModelName}/model.joblib`
  - **Global artifacts**: `models/_global/scaler.joblib`, `models/_global/feature_columns.json`
  - **Results**: `results/{ModelName}/`
    - `metrics.json` - Accuracy, precision, recall, F1, AUC
    - `classification_report.txt` - Detailed performance report
    - `confusion_matrix.png` - Confusion matrix visualization
    - `roc_curve.png` - ROC curve plot
    - `feature_importance.csv/.png` - Feature importance analysis (when supported)

#### `TrainAllModels_DL.py`
- **Purpose**: Trains deep learning models (both tabular and text-based)
- **Inputs**: `features_extracted.csv`
- **Models Trained**:
  - **Tabular**: DL-MLP, DL-FTTransformer (uses numeric features)
  - **Text**: DL-CharCNN, DL-CharTransformer (uses URL strings)
- **Outputs**:
  - **Models**: `models/{ModelName}/model.pt`
  - **Global artifacts**: 
    - `models/_global/url_tokenizer.json` - Text tokenization config
    - `models/_global/split_indices.json` - Train/val/test split indices
  - **Results**: `results/{ModelName}/`
    - `metrics.json` - Performance metrics
    - `classification_report.txt` - Detailed classification report
    - `confusion_matrix.png` - Confusion matrix
    - `roc_curve.png` - ROC curve
    - `test_with_preds.csv` - Test predictions with URLs

#### `TrainAllFeatureDefensiveModels.py`
- **Purpose**: Trains 3-class models (0=benign, 1=malicious, 2=adversarial)
- **Inputs**: `features_base3_strict_cap.csv`
- **Models Trained**: Same as ML script but for 3-class classification
- **Outputs**:
  - **Models**: `models_base3/{ModelName}/model.joblib` or `model.pt`
  - **Global artifacts**: `models_base3/_global/` (scaler, features, split info)
  - **Results**: `results_base3/{ModelName}/`
    - `metrics.json` - 3-class performance metrics
    - `classification_report.txt` - Multi-class report
    - `confusion_matrix.png` - 3-class confusion matrix
    - `roc_curve.png` - Multi-class ROC curves
    - `feature_importance.csv/.png` - Feature importance (when supported)
    - `train_log.txt` - Training logs

### **Adversarial Evaluation Scripts**

#### `attackModels.py`
- **Purpose**: Tests all trained models against adversarial attacks
- **Attacks Implemented**:
  - **FGSM** (Fast Gradient Sign Method)
  - **PGD** (Projected Gradient Descent)
  - **White-box attacks** for PyTorch and linear models
  - **Black-box attacks** via surrogate models for tree-based models
  - **HotFlip attacks** for character-based text models
- **Outputs**: `results_evaluation/{ModelName}/`
  - `adv_NAT/` - Natural (unattacked) performance
    - `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`
  - `adv_FGSM/` - FGSM attack results
    - `attack_config.json` - Attack configuration
    - `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`
  - `adv_PGD/` - PGD attack results
    - `attack_config.json` - Attack configuration
    - `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`

### **Defense System Scripts**

#### `train_DefensiveLightGBM.py`
- **Purpose**: Trains an adversarial detector to identify attacked samples
- **Inputs**: `features_adversarial_defense_dataset.csv`
- **Outputs**: 
  - **Models**: `models/Defense-LGBM/`
    - `model.joblib` - Trained LightGBM detector
    - `calibrator.joblib` - Probability calibrator
  - **Results**: `results_defense_features/Defense-LGBM/`
    - `candidate_summary.json` - Model comparison summary
    - `metrics_val.json`, `metrics_test.json` - Validation and test performance
    - `classification_report_val.txt`, `classification_report_test.txt`
    - `confusion_matrix_val.png`, `confusion_matrix_test.png`
    - `roc_curve_val.png`, `roc_curve_test.png`
    - `feature_importance_top30.png` - Top feature importance
    - `params_used.json` - Model parameters

#### `train_DefensiveCharCNN.py`
- **Purpose**: Trains character-based adversarial detector
- **Outputs**: `models/Defense-CharCNN/`
  - `model.pt` - Trained CharCNN detector

#### `defence_FeatureModels.py`
- **Purpose**: Evaluates defense effectiveness on feature-based models
- **Process**: 
  1. Crafts adversarial examples against base models
  2. Uses defense detector to filter suspicious samples
  3. Evaluates base model performance on accepted samples
- **Outputs**: `results_defense_feat_stream/{ModelName}/`
  - `_common/` - Shared cache and histograms
    - `p_adv_nat_full_*.npy` - Cached NAT probability scores
    - `p_adv_nat_hist.png` - NAT probability distribution
  - `NAT_pure/` - Performance on natural samples only
    - `metrics.json` - Pure natural performance
    - `base_confusion_all_nat.png`, `base_confusion_accepted_nat.png`
    - `base_roc_all_nat.png`, `base_roc_accepted_nat.png`
    - `classification_report_all_nat.txt`, `classification_report_accepted_nat.txt`
  - `mixed_FGSM/`, `mixed_PGD/` - Defense performance against different attacks
    - `metrics.json` - Rich defense metrics
    - `detector_confusion_tau.png` - Detector confusion at threshold
    - `detector_roc_nat_vs_adv.png` - Detector ROC curve
    - `base_confusion_accepted_nat.png`, `base_roc_accepted_nat.png`
    - `classification_report_accepted_nat.txt`
    - `p_adv_nat_hist.png`, `p_adv_adv_hist.png`, `p_adv_adv_hist_small.png`

#### `defence_FeatureModels3Class.py`
- **Purpose**: 3-class defense evaluation (benign, malicious, adversarial)
- **Outputs**: `results_defence_features_3class/{ModelName}/`
  - `_common/` - Shared cached NAT probabilities
  - `mixed_FGSM/`, `mixed_PGD/` - 3-class defense results
    - `metrics_before_defense.json` - Performance without defense
    - `metrics_after_defense.json` - Performance with defense
    - `confusion_no_def.png`, `confusion_after_def.png` - 3-class confusion matrices
    - `confusion_nat_no_def.png`, `confusion_nat_after_def.png` - NAT-only confusion
    - `roc_ovr_no_def.png`, `roc_ovr_after_def.png` - Multi-class ROC curves
    - `classification_report_no_def.txt`, `classification_report_after_def.txt`

#### `defense_CharacterModels.py`
- **Purpose**: Evaluates defense on character-based deep learning models
- **Outputs**: `results_defense_char_stream/{ModelName}/`
  - `_common/` - Shared histograms
  - `NAT_pure/` - Natural performance
    - `metrics.json`
    - `base_confusion_all_nat.png`, `base_confusion_accepted_nat.png`
    - `base_roc_all_nat.png`, `base_roc_accepted_nat.png`
    - `classification_report_all_nat.txt`, `classification_report_accepted_nat.txt`
  - `mixed_FGSM/`, `mixed_PGD/` - Defense against HotFlip attacks
    - `metrics.json` - Rich defense metrics
    - `detector_confusion_tau.png`, `detector_roc_nat_vs_adv.png`
    - `base_confusion_accepted_nat.png`, `base_roc_accepted_nat.png`
    - `classification_report_accepted_nat.txt`
    - `p_adv_nat_hist.png`, `p_adv_adv_hist.png`, `p_adv_adv_hist_small.png`

### **Defense Data Preparation Scripts**

#### `prep_defensive_dataset_features.py`
- **Purpose**: Creates training data for the adversarial detector
- **Process**:
  1. Loads trained base models
  2. Crafts adversarial examples using FGSM, PGD, C&W, and noise attacks
  3. Labels samples as natural (0) or adversarial (1)
- **Outputs**: `features_adversarial_defense_dataset.csv`
  - Contains original features plus adversarial label
  - Used to train the defense detector

#### `prep_defensive_dataset_characters.py`
- **Purpose**: Creates character-based adversarial training data
- **Outputs**: Character-based defense datasets for training CharCNN detector

#### `prep_defence_dataset_features_for_training.py`
- **Purpose**: Prepares and preprocesses defense datasets for training
- **Outputs**: Processed defense training datasets

#### `remove_dups_defensive.py`
- **Purpose**: Removes duplicate entries from defense datasets
- **Outputs**: Cleaned defense datasets without duplicates

---

## 📂 Directory Structure

### **`models/`** - Binary Model Storage
```
models/
├── _global/                    # Shared artifacts
│   ├── scaler.joblib          # Feature scaler