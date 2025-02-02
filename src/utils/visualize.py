import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(data, column, title=None):
    """Plot distribution of a column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column])
    if title:
        plt.title(title)
    plt.show()

def plot_correlation_matrix(data):
    """Plot correlation matrix of numerical columns."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
