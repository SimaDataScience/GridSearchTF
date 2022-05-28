"""Create and save training, validation, and test splits."""
import os
import pandas as pd

from utilities.constant import ROOT_DIR

def create_data_splits(data_path: str, test_size: float=0.01, validation_size: float=0.02):
    """_summary_

    Args:
        data_path (str): _description_
        test_size (float, optional): _description_. Defaults to 0.01.
        validation_size (float, optional): _description_. Defaults to 0.02.
    """
    # Load data.
    df = pd.read_csv(data_path)

    # Create training, validation, and test sets.
    df_test = df.sample(frac=test_size, random_state=49)

    df_ = df.drop(df_test.index)

    df_val = df_.sample(frac=validation_size, random_state=49)
    df_train = df_.drop(df_val.index).sample(frac=1.0, random_state=49)

    train_path = os.path.join(ROOT_DIR, 'data', 'data-train.csv')
    validation_path = os.path.join(ROOT_DIR, 'data', 'data-validation.csv')
    test_path = os.path.join(ROOT_DIR, 'data', 'data-test.csv')

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(validation_path, index=False)
    df_test.to_csv(test_path, index=False)
