#!/bin/bash

# Update pip
echo "Updating pip..."
pip3 install --upgrade pip

# Core Machine Learning
echo "Installing core machine learning libraries..."
pip3 install scikit-learn tensorflow torch keras xgboost lightgbm

# Data Handling & Preprocessing
echo "Installing data handling libraries..."
pip3 install pandas numpy scipy

# Visualization
echo "Installing visualization libraries..."
pip3 install matplotlib seaborn plotly

# Experiment Tracking & Model Management
echo "Installing experiment tracking libraries..."
pip3 install mlflow tensorboard

# Natural Language Processing (NLP)
echo "Installing NLP libraries..."
pip3 install nltk spacy transformers

# Image Processing
echo "Installing image processing libraries..."
pip3 install opencv-python pillow

# Web Scraping & APIs
echo "Installing web scraping libraries..."
pip3 install requests beautifulsoup4

# Utilities
echo "Installing utility libraries..."
pip3 install tqdm joblib

# Verify installations
echo "Verifying installations..."
pip3 list

echo "All essential ML libraries installed successfully!"