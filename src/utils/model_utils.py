from sklearn.model_selection import train_test_split
from ..config import RANDOM_SEED

def split_data(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    return model.score(X_test, y_test)
