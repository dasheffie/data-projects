import category_encoders as ce
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

import category_encoders as ce
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd

def weighted_mean_k_fold_target_encoding(df, encode_col: str, target_col: str, k=5, smoothing=10, kf_shuffle: bool=True, kf_shuffle_random_state: int=None, stratified: bool=False):
    """
    Perform k-fold target encoding using category_encoders with weighted mean smoothing.

    Parameters:
    - df: DataFrame containing the data.
    - encode_col: Column containing the categorical feature to be encoded.
    - target_col: Column containing the target variable.
    - k: Number of splits for K-Fold or Stratified K-Fold cross-validation.
    - smoothing: Smoothing parameter (AKA, 'm') for TargetEncoder. (n * option mean + m * overall mean) / (n + m)
    - kf_shuffle: Whether to shuffle the K-Fold or Stratified K-Fold splits.
    - kf_shuffle_random_state: Random state for K-Fold or Stratified K-Fold splits.
    - suffix: Suffix to add to the target encoded column name.
    - stratified: Whether to use Stratified K-Fold instead of regular K-Fold.

    Returns:
    - A NumPy array with the encoded values.
    """
    df = df.copy()
    encoded_values = np.zeros(len(df))  # Store encoded values

    # Choose between KFold and StratifiedKFold
    if stratified:
        kf = StratifiedKFold(n_splits=k, shuffle=kf_shuffle, random_state=kf_shuffle_random_state)
    else:
        kf = KFold(n_splits=k, shuffle=kf_shuffle, random_state=kf_shuffle_random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df[encode_col] if stratified else None)):
        train_data, val_data = df.iloc[train_idx], df.iloc[val_idx]

        # Initialize and fit the TargetEncoder
        encoder = ce.TargetEncoder(cols=[encode_col], smoothing=smoothing)
        encoder.fit(train_data[encode_col], train_data[target_col])

        # Transform validation data
        val_data_encoded = encoder.transform(val_data[encode_col])
        encoded_values[val_idx] = val_data_encoded[encode_col]

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

def weighted_average_encoding(df, numeric_col: str, categorical_cols: list, smoothing: int = 10):
    """
    Calculate a weighted average for combinations of values in multiple categorical columns based on a numeric column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_col (str): The name of the numeric column to average.
    - categorical_cols (list): A list of categorical column names used to weight the average.
    - smoothing (int): The regularization parameter (default: 10).

    Returns:
    - pd.Series: A series containing the weighted average for each combination of categorical columns.
    """
    df = df.copy()

    # Calculate the overall mean of the numeric column
    overall_mean = df[numeric_col].mean()

    # Calculate the mean and count for each combination of categorical columns
    group_stats = df.groupby(categorical_cols)[numeric_col].agg(['mean', 'count']).reset_index()
    group_stats.columns = categorical_cols + ['option_mean', 'n']

    # Calculate the weighted average
    group_stats['weighted_mean'] = ((group_stats['n'] * group_stats['option_mean'] + smoothing * overall_mean) / (group_stats['n'] + smoothing))

    # Map the weighted mean back to the original dataframe
    weighted_avg_series = (df.merge(group_stats[categorical_cols + ['weighted_mean']], on=categorical_cols, how='left')['weighted_mean'])

    # Fill NAs in the numeric column with the overall mean
    weighted_avg_series.fillna(overall_mean, inplace=True)

    return weighted_avg_series
