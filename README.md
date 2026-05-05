# German Credit Neural Network

This repository contains a PyTorch-based neural network workflow for binary credit risk classification on the German Credit dataset. The project trains a feed-forward classifier, evaluates model performance with common classification metrics, and saves precision-recall / ROC curve outputs for review.

## Project Overview

The main non-XAI script trains a neural network to classify customers as good or bad credit risks using the processed German Credit dataset. It performs preprocessing with `scikit-learn`, trains the model with PyTorch, and reports metrics such as accuracy, F1 score, AUC-PR, ROC-AUC, and a confusion matrix.

An additional XAI script is included for explainability experiments with SHAP/Shapash-style tooling.

## Repository Structure

```text
.
├── configs/
│   └── german_credit/
│       └── params_neuralnet.json
├── data/
│   └── raw/
│       └── german_credit.csv
├── src/
│   ├── german_credit_neuralnet_without_xai.py
│   └── german_credit_neuralnet_with_xai.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Main Features

- Trains a binary neural network classifier with PyTorch
- Uses a configurable architecture and training parameters
- Standardizes numeric features before training
- Splits the dataset into train and test sets
- Computes F1 score, accuracy, confusion matrix, AUC-PR, and ROC-AUC
- Generates precision-recall and ROC curve visualizations
- Includes a separate explainability-oriented script for XAI experiments

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the non-XAI training workflow from the `src` directory:

```bash
cd src
python german_credit_neuralnet_without_xai.py
```

The script reads the processed dataset from `data/raw/german_credit.csv`, trains the model, prints evaluation results, and writes generated outputs such as plots and performance CSV files.

## Notes

The XAI script imports additional packages such as `shap` and `shapash`, which are not included in the minimal `requirements.txt` for the non-XAI workflow.
