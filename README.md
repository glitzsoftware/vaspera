# Vaspera: AI-Driven Antimicrobial Resistance Prediction System

## Overview
Vaspera is a clinical decision support system designed to predict antibiotic resistance phenotypes and optimize drug dosages using machine learning. The system helps reduce trial-and-error antibiotic prescriptions and provides interpretable treatment recommendations to clinicians.

## Key Features
- Predicts antibiotic resistance (Resistant/Susceptible/Intermediate) using supervised classification
- Recommends optimal drug dosages via supervised regression
- Discovers latent drug-bacteria patterns using unsupervised learning
- Provides predictions via API and interactive dashboard
- Includes comprehensive MLOps setup with experiment tracking

## Tech Stack
- **Core ML**: Python, scikit-learn, PyTorch
- **API & Dashboard**: FastAPI, Plotly Dash
- **MLOps**: MLflow

## Project Structure
```
.
├── data/               # Data storage (not tracked in git)
├── models/            
│   ├── mlruns/        # MLflow experiment tracking
│   └── training_logs/ # Model training logs
├── notebooks/
│   ├── data_processing.ipynb  # Data preprocessing and analysis
│   └── xgboost.ipynb         # XGBoost model training
├── src/
│   ├── utils/         # Utility functions
│   ├── config.py      # Configuration settings
│   ├── pipeline.py    # Data processing pipeline
│   └── settings.py    # Application settings
├── install.sh         # Installation script
├── requirements.txt   # Python dependencies
└── technical-design.md # Detailed technical documentation
```

## Installation
1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage
1. **Data Processing**:
   - Use the `notebooks/data_processing.ipynb` notebook to preprocess your data
   - Follow the data pipeline defined in `src/pipeline.py`

2. **Model Training**:
   - Use the `notebooks/xgboost.ipynb` notebook for training the resistance prediction model
   - All experiments are tracked using MLflow in the `models/mlruns` directory

3. **Model Deployment**:
   - The system is containerized using Docker for easy deployment
   - API endpoints are available for resistance prediction and dosage optimization

## Development
- The project follows a modular architecture for easy expansion
- MLflow is used for experiment tracking and model versioning
- Code quality is maintained through automated testing and CI/CD

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Authors
@@bemijonathan @@Aymen006