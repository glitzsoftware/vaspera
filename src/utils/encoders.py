import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Union, List
from joblib import Parallel, delayed


class CategoricalEncoders:
    """for encoding categorical variables"""

    def __init__(self) -> None:
        pass

    def label_encode(self, data: pd.Series) -> tuple[pd.Series, LabelEncoder]:
        encoder = LabelEncoder()
        encoded_data = pd.Series(encoder.fit_transform(data), index=data.index)
        return encoded_data, encoder

    def frequency_encode(self, data: pd.Series) -> tuple[pd.Series, Dict]:
        frequency_map = data.value_counts(normalize=True).to_dict()
        encoded_data = data.map(frequency_map)
        return encoded_data, frequency_map

    def target_encode(self, cat: pd.Series, targets: pd.DataFrame, min_samples: int = 10, smoothing: int = 20):
        """
        Applies target encoding for each column in the targets DataFrame.

        Parameters:
            cat (pd.Series): The categorical feature.
            targets (pd.DataFrame): One or more target columns to encode.

        Returns:
            encoded_data (pd.DataFrame): DataFrame with new target-encoded columns.
            encoding_dict (dict): Dictionary mapping target column names to their encoding mappings.
        """
        encoded_data = pd.DataFrame(index=cat.index)
        encoding_dict = {}

        for col in targets.columns:
            encoded_col, encoding = self.target_encode_series(
                cat, targets[col], min_samples, smoothing)
            encoded_data[f'{cat.name}_te_{col}'] = encoded_col
            encoding_dict[col] = encoding

        return encoded_data, encoding_dict

    def target_encode_series(self, cat: pd.Series, target: pd.Series, min_samples: int = 10, smoothing: int = 20):
        """
        Computes a smoothed target encoding for a single target column.

        Parameters:
            cat (pd.Series): The categorical feature.
            target (pd.Series): The target values corresponding to each row.
            min_samples (int): Minimum count threshold for a category to be trusted.
            smoothing (int): Smoothing factor for the mean.

        Returns:
            encoded (pd.Series): The encoded values mapped back to the original series.
            encoding_dict (dict): Mapping from category to smoothed mean.
        """
        # Global mean of the target column
        global_mean = target.mean()

        # Compute per-category statistics in one shot
        stats = target.groupby(cat).agg(mean='mean', count='count')

        # Apply smoothing: higher counts get closer to the raw mean, lower counts are shrunk toward global_mean
        stats['smoothed'] = (stats['mean'] * stats['count'] +
                             global_mean * smoothing) / (stats['count'] + smoothing)

        # For categories with few samples, assign the global mean
        stats.loc[stats['count'] < min_samples, 'smoothed'] = global_mean

        # Map the smoothed means back to the original categorical series
        encoded = cat.map(stats['smoothed'])

        return encoded, stats['smoothed'].to_dict()

    def one_hot_encode(self, data: pd.Series, max_categories: int = None) -> pd.DataFrame:
        """ this returns the already encoded data """
        if max_categories and len(data.unique()) > max_categories:
            # Keep only top categories
            top_categories = data.value_counts().nlargest(max_categories).index
            modified_data = data.copy()
            modified_data[~modified_data.isin(top_categories)] = 'other'
            return pd.get_dummies(modified_data, prefix=data.name, dtype=int)

        return pd.get_dummies(data, prefix=data.name, dtype=int)

    def encode_column(self, data: pd.Series,
                      method: str = 'label',
                      target: pd.Series = None,
                      max_categories: int = None,
                      min_samples: int = 10,
                      smoothing: int = 20) -> Union[pd.Series, pd.DataFrame]:
        if method == 'label':
            encoded_data, _ = self.label_encode(data)
            return encoded_data
        elif method == 'frequency':
            encoded_data, _ = self.frequency_encode(data)
            return encoded_data
        elif method == 'target':
            if target is None:
                raise ValueError(
                    "Target series must be provided for target encoding")
            encoded_data, _ = self.target_encode(
                data, target, min_samples, smoothing)
            return encoded_data
        elif method == 'onehot':
            return self.one_hot_encode(data, max_categories)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
