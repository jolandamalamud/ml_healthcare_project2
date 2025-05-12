# README

## Overview

This repository contains code for **Part 1** of ML4H **Project 2**, focusing on **interpretable and explainable classification** for a Heart Disease Prediction dataset. The dataset is from Kaggle (aggregated from the UCI ML Repository). All code for each question resides in corresponding **.ipynb** notebooks.

---

## Repository Structure (Part 1)

├── Q1 │ ├── 01_data_exploration_and_preprocessing.ipynb │ 
	└── preprocessor.pkl
		├── Q2 │ ├── 02_logistic_lasso.ipynb │├── logreg_l1.pkl │
			└──├── Q3 │ ├── 03_mlp_shap.ipynb │ ├── best_mlp_pipe.pkl │
				└──├── Q4 │ ├── 04_nam_model.ipynb │ ├── nam_pipe.pkl 

### Q1 — Data Exploration & Preprocessing
- **Notebook**: `01_data_exploration_and_preprocessing.ipynb`  
  - Explores data (EDA, histograms, correlation).  
  - Cleans missing/impossible numeric values, does KNN-imputation in scaled space.  
  - One-hot encodes categoricals, re-scales numerics, and saves the final pipeline to `preprocessor.pkl`.

### Q2 — Logistic Lasso Regression
- **Notebook**: `02_logistic_lasso.ipynb`  
  - Loads `preprocessor.pkl` (so data transformations are consistent).  
  - Trains a logistic model with L1 (lasso) regularization.  
  - Saves the resulting pipeline to `logreg_l1.pkl`.

### Q3 — MLP + SHAP
- **Notebook**: `03_mlp_shap.ipynb`  
  - Loads `preprocessor.pkl`.  
  - Trains a Multi-Layer Perceptron, does Bayesian hyperparameter tuning.  
  - Uses **SHAP** for local and global explanations.  
  - Saves pipeline as `best_mlp_pipe.pkl`.

### Q4 — Neural Additive Model (NAM)
- **Notebook**: `04_nam_model.ipynb`  
  - Implemented in TensorFlow/Keras, one small sub-network per feature.  
  - Saves the final pipeline to `nam_pipe.pkl`.  
  - **Note**: Q4 can be run in a **separate Python environment** if needed. The `requirements.txt` (or alternative environment file) is provided with the necessary packages (including TensorFlow).

---

## Data Requirements

Place the CSV files in a `data/` folder (or update paths in notebooks):