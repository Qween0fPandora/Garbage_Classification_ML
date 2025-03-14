# Garbage_Classification_ML

This repository contains a comprehensive traditional machine learning solution for classifying garbage images into five categories: battery, clothes, glass, paper, and shoes.

## Features

- **Multiple Feature Extraction Methods**: HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns)
- **Three Classifier Options**: SVM, Random Forest, and KNN with optimized parameters
- **Data Balancing and Augmentation**: Techniques to handle class imbalance
- **Model Optimization**: Hyperparameter tuning with cross-validation
- **Performance Evaluation**: Detailed metrics and visualizations
- **Testing Framework**: Support for complex real-world scenes
- **Model Persistence**: Save and load trained models

## Requirements

See [requirements.txt](requirements.txt) for a complete list of dependencies.

## Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/garbage-classification.git
cd garbage-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python garbage_classification.py
```

4. Select an operation mode when prompted:
   - **Option 1**: Run baseline models (default parameters)
   - **Option 2**: Run optimized SVM model
   - **Option 3**: Run optimized Random Forest model
   - **Option 4**: Run optimized KNN model
   - **Option 5**: Test previously trained models on complex dataset

5. Provide paths to your datasets when prompted:
   - **Main dataset**: Should contain 5 subdirectories named 'battery', 'clothes', 'glass', 'paper', 'shoes'
   - **Complex dataset** (optional): Contains challenging real-world images for model testing

