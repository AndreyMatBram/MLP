# MLP Implementation with NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-MLP-orange)

This repository contains implementations of a Multi-Layer Perceptron (MLP) using only NumPy in Python. Three different experiments were conducted to demonstrate the MLP's capabilities on various datasets.

## Table of Contents
- [Experiments](#experiments)
  - [1. Iris Dataset](#1-iris-dataset)
  - [2. Wine Dataset](#2-wine-dataset)
  - [3. Car Evaluation Dataset](#3-car-evaluation-dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)

## Experiments

### 1. Iris Dataset

**Dataset Description**:  
Classic Iris flower dataset with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica).

**Preprocessing**:
- Output classes converted to one-hot encoding
- Data shuffled before splitting
- No feature scaling applied

**Network Architecture**:
- Input layer: 4 neurons
- Hidden layer: 4 neurons
- Output layer: 3 neurons

**Hyperparameters**:
- Learning rate: 0.017
- Activation function: Sigmoid
- Epochs: 1000
- Data split: 60% train, 20% validation, 20% test

**Results**:
- Accuracy: >94% on test set
- The model showed excellent performance in distinguishing between the three flower species

### 2. Wine Dataset

**Dataset Description**:  
Wine recognition data with 13 chemical features and 3 classes (different cultivars from Italy).

**Preprocessing**:
- All 13 features were kept
- Output classes converted to one-hot encoding
- Data shuffled before splitting

**Network Architecture**:
- Input layer: 13 neurons
- Hidden layer: 4 neurons
- Output layer: 3 neurons

**Hyperparameters**:
- Learning rate: 0.017
- Activation function: Sigmoid
- Epochs: 1000
- Data split: 60% train, 20% validation, 20% test

**Results**:
- Accuracy: >97% on test set
- The model performed exceptionally well, demonstrating MLP's capability with higher-dimensional data

### 3. Car Evaluation Dataset

**Dataset Description**:  
Car evaluation dataset with 6 categorical attributes and 4 classes (unacceptable, acceptable, good, very good).

**Preprocessing**:
- All categorical features converted using one-hot encoding
- Output classes converted to one-hot encoding
- Significant class imbalance (unacc >900 of 1728 instances)

**Network Architecture**:
- Input layer: >20 neurons (after one-hot encoding)
- Hidden layer: 6 neurons
- Output layer: 4 neurons

**Hyperparameters**:
- Learning rate: 0.017
- Activation function: Sigmoid
- Epochs: 1000
- Data split: 60% train, 20% validation, 20% test

**Results**:
- Apparent accuracy: 93%
- Poor performance on minority classes
- Demonstrates the challenges of imbalanced datasets

## Requirements

To run these implementations, you need:
- Python 3.8 or higher
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mlp-numpy-implementation.git
cd mlp-numpy-implementation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Each experiment has its own script:

```bash
# Run Iris experiment
python mlp_iris.py

# Run Wine experiment
python mlp_wine.py

# Run Car Evaluation experiment
python mlp_car.py
```

The scripts will:
1. Load and preprocess the data
2. Train the MLP model
3. Evaluate on test set
4. Display confusion matrices and accuracy metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **Author**: Andrey Matheus Brambilla
- **Acknowledgments**: 
  - UCI Machine Learning Repository for the datasets
  - NumPy community for the foundational library

---

This README provides comprehensive documentation for your MLP implementation project, making it easy for others to understand and reproduce your experiments. The badge at the top adds visual appeal and quick reference to key information.
