import category_encoders as ce
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def weighted_mean_k_fold_target_encoding(df, target_encode_col: str, target_col: str, k, smoothing):
    """
    Perform k-fold target encoding using category_encoders with weighted mean smoothing.
    
    Parameters:
    - df: DataFrame containing the data.
    - target_encode_col: Column containing the categorical feature to be target encoded.
    - target_col: Column containing the target variable.
    - k: Number of splits for K-Fold cross-validation.
    - smoothing: Smoothing parameter (AKA, 'm') for TargetEncoder. (n * option mean + m * overall mean) / (n + m)
    
    Returns:
    - Encoded values as a NumPy array.
    """
    df = df.copy()
    encoded_values = np.zeros(len(df))  # Store encoded values

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train_data, val_data = df.iloc[train_idx], df.iloc[val_idx]

        # Initialize and fit the TargetEncoder
        encoder = ce.TargetEncoder(cols=[target_encode_col], smoothing=smoothing)
        encoder.fit(train_data[target_encode_col], train_data[target_col])

        # Transform validation data
        val_data_encoded = encoder.transform(val_data[target_encode_col])
        encoded_values[val_idx] = val_data_encoded[target_encode_col]

    return encoded_values

def clean_and_split_string(element, fill_nan=True):
    """
    Cleans and splits a single element by removing NaN values (if specified),
    splitting on commas, and stripping whitespace from each item.

    Parameters:
    element (str or NaN): The element to process.
    fill_nan (bool): If True, replace NaN values with an empty string before processing.

    Returns:
    list: A list of cleaned and split items.
    """
    if pd.isna(element) and fill_nan:
        element = ""
    if isinstance(element, str):
        return [item.strip() for item in element.split(",") if item.strip()]
    return []

# Example usage:
# df['col'] = df['col'].apply(lambda x: clean_and_split_element(x))

import pandas as pd

def weighted_average_encoding(df, numeric_col: str, categorical_col: str, m: int =10):
    """
    Calculate a weighted average for each value of the categorical column based on a numeric column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_col (str): The name of the numeric column to average.
    - categorical_col (str): The name of the categorical column used to weight the average.
    - m (float): The regularization parameter (default: 10).

    Returns:
    - pd.Series: A series containing the weighted average for each value of the categorical column.
    """
    df = df.copy()

    # Calculate the overall mean of the numeric column
    overall_mean = df[numeric_col].mean()

    # Calculate the mean and count for each value of the categorical column
    group_stats = df.groupby(categorical_col)[numeric_col].agg(['mean', 'count']).reset_index()
    group_stats.columns = [categorical_col, 'option_mean', 'n']

    # Calculate the weighted average
    group_stats['weighted_mean'] = (
        (group_stats['n'] * group_stats['option_mean'] + m * overall_mean) /
        (group_stats['n'] + m)
    )

    # Map the weighted mean back to the original dataframe
    weighted_avg_series = df[categorical_col].map(group_stats.set_index(categorical_col)['weighted_mean'])
    
    # # Fill NAs in the numeric column with the overall mean
    weighted_avg_series.fillna(overall_mean, inplace=True)

    return weighted_avg_series
