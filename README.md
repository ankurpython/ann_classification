# ANN Classification Project

## Overview
This repository, `ann_classification`, contains an implementation of an Artificial Neural Network (ANN) for classification tasks using Python. The project demonstrates how to build, train, and evaluate an ANN model to solve classification problems, such as predicting customer churn, using popular libraries like TensorFlow, Keras, and scikit-learn. The codebase includes data preprocessing, model creation, training, and evaluation steps, making it a valuable resource for learning and applying ANN techniques.

## Features
- **Data Preprocessing**: Loads, cleans, and prepares datasets for ANN training.
- **ANN Model**: Implements a feedforward neural network with customizable layers, neurons, and activation functions.
- **Classification**: Supports binary or multi-class classification tasks (e.g., churn prediction).
- **Evaluation**: Provides metrics like accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization**: Includes plots for training/validation accuracy and loss over epochs.

## Dataset
The project uses the **Churn Modelling** dataset (or a similar dataset), which contains features like credit score, geography, gender, age, tenure, balance, and more to predict customer churn. The dataset is not included in the repository but can be sourced from platforms like Kaggle or UCI Machine Learning Repository.

## Requirements
To run the code, ensure you have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.x
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ankurpython/ann_classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ann_classification
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the Dataset**:
   - Place the dataset (e.g., `Churn_Modelling.csv`) in the project’s root directory or update the file path in the code.
   - The dataset should include features and a target variable for classification.

2. **Run the Code**:
   - Open the Jupyter notebook (`notebook.ipynb`) or Python script (`ann.py`):
     ```bash
     jupyter notebook notebook.ipynb
     ```
     or
     ```bash
     python ann.py
     ```

3. **Steps in the Code**:
   - **Data Preprocessing**: Loads the dataset, handles missing values, encodes categorical variables, and splits data into training and test sets.
   - **Model Building**: Constructs an ANN with an input layer, hidden layers (using ReLU activation), and an output layer (using sigmoid for binary classification or softmax for multi-class).
   - **Training**: Trains the model using the Adam optimizer and binary or categorical cross-entropy loss.
   - **Evaluation**: Computes metrics like accuracy and generates a confusion matrix.
   - **Visualization**: Plots training/validation accuracy and loss.

4. **Example Output**:
   - Accuracy, precision, recall, and F1-score printed to the console.
   - Plots of model performance saved as images (e.g., `accuracy_plot.png`, `loss_plot.png`).

## Project Structure
```plaintext
ann_classification/
│
├── data/                   # Folder for dataset (not included)
├── notebook.ipynb          # Jupyter notebook with the full workflow
├── ann.py                 # Main Python script for ANN implementation
├── requirements.txt        # List of dependencies
├── README.md              # This file
├── plots/                  # Folder for saving output plots (e.g., accuracy, loss)
```

