"""Returns transformed training, validation, and testing sets."""
import os
import numpy as np
import pandas as pd

from utilities.constant import ROOT_DIR

def load_all_datasets(label_name: str, sample_frac: float, transformer=None):
    """_summary_

    Args:
        label_name (str): _description_
        sample_frac (float): _description_
        transformer (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    train_path = os.path.join(ROOT_DIR, 'data', 'data-train.csv')
    val_path = os.path.join(ROOT_DIR, 'data', 'data-validation.csv')
    test_path = os.path.join(ROOT_DIR, 'data', 'data-test.csv')

    df_train = pd.read_csv(train_path).sample(frac=sample_frac)
    df_val = pd.read_csv(val_path).sample(frac=sample_frac)
    df_test = pd.read_csv(test_path).sample(frac=sample_frac)

    X_train = df_train.drop(columns=[label_name]).copy()
    y_train = df_train[label_name].copy()

    X_val = df_val.drop(columns=[label_name]).copy()
    y_val = df_val[label_name].copy()

    X_test = df_test.drop(columns=[label_name]).copy()
    y_test = df_test[label_name].copy()

    if transformer:
        transformer.fit(X_train, y_train)

        X_train = transformer.transform(X_train)
        X_val = transformer.transform(X_val)
        X_test = transformer.transform(X_test)

    # Transform label.
    y_train = np.log(y_train)
    y_val = np.log(y_val)
    y_test = np.log(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_single_dataset(label_name: str, dataset_name: str='train'):
    """_summary_

    Args:
        label_name (str): _description_
        dataset_name (str, optional): _description_. Defaults to 'train'.

    Returns:
        _type_: _description_
    """
    # Load training data and fit transformer.
    train_path = os.path.join(ROOT_DIR, 'data', 'data-train.csv')
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=[label_name])
    custom_transformer.fit(X_train)

    # Check for a likely mistake.
    if dataset_name == 'val':
        dataset_name = 'validaiton'

    # Load desired dataset.
    data_path = os.path.join(ROOT_DIR, 'data', f'data-{dataset_name}.csv')
    df = pd.read_csv(data_path)

    # Separate features and labels, transform data.
    X = df.drop(columns=[label_name]).copy()
    X_scaled = custom_transformer.transform(X)
    y = df[label_name].copy()

    return X_scaled, y
