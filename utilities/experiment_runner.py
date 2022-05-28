import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utilities.constant import ROOT_DIR
from utilities.dataset_splitter import create_data_splits
from utilities.dataset_loader import  load_all_datasets

class TensorflowExperiment:
    """_summary_
    """
    def __init__(self, save_name, model_builder, transformer=None, create_experiment_data: bool=False, data_path: str='') -> None:
        """_summary_

        Args:
            save_name (_type_): _description_
            model_builder (_type_): _description_
            transformer (_type_, optional): _description_. Defaults to None.
            create_experiment_data (bool, optional): _description_. Defaults to False.
            data_path (str, optional): _description_. Defaults to ''.
        """
        # Results container.
        self.df_results = pd.DataFrame()
        self.save_name = save_name
        self.model_builder = model_builder
        self.transformer = transformer

        # Create directories for results.
        experiment_folder = os.path.join(ROOT_DIR, 'experiments', self.save_name)
        Path(experiment_folder).mkdir(parents=True, exist_ok=True)

        images_folder = os.path.join(ROOT_DIR, 'experiments', self.save_name, 'images')
        Path(images_folder).mkdir(parents=True, exist_ok=True)

        if create_experiment_data:
            create_data_splits(data_path=data_path)

    def run_experiment(
        self,

        model_params:dict,
        label_name: str,

        sample_frac: float=1,
        batch_size: int=32,
        max_epochs: int=5,
        callback_threshold: float=0.001,
        patience: int=20,
        sleep_time: int=0
    ):
        """_summary_

        Args:
            sample_frac (float): _description_
            batch_size (int): _description_
            max_epochs (int): _description_
            callback_threshold (float): _description_
            patience (int): _description_
            sleep_time (int): _description_
            model_params (dict): _description_
            label_name (str): _description_
        """
        # Load data.
        X_train, X_val, X_test, y_train, y_val, y_test = load_all_datasets(
            label_name=label_name,
            sample_frac=sample_frac,
            transformer=self.transformer
        )

        # Create dictionaries containing every combination from input arguments.
        parameter_grid = ParameterGrid(model_params)

        # Run experiment and save results.
        for trial_number, parameters in tqdm(enumerate(parameter_grid)):
            print(f'Model parameters: {parameters}')

            experiment_results = build_and_evaluate_model(
                self.model_builder,

                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,

                batch_size,
                max_epochs,
                callback_threshold,
                patience,

                params=parameters,

                experiment_number=trial_number,
                save_name=self.save_name
            )

            self.df_results = pd.concat([self.df_results, pd.DataFrame([experiment_results])])

            time.sleep(sleep_time)

        # Save results.
        file_name = os.path.join(ROOT_DIR, 'experiments', self.save_name, self.save_name + '-results.csv')

        self.df_results.to_csv(file_name, index_label='model_number')

def build_and_evaluate_model(
    build_model,

    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,

    batch_size,
    max_epochs,
    callback_threshold,
    patience,
    params,
    experiment_number,
    save_name
):
    """_summary_

    Args:
        build_model (_type_): _description_
        X_train (_type_): _description_
        X_val (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_val (_type_): _description_
        y_test (_type_): _description_
        batch_size (_type_): _description_
        max_epochs (_type_): _description_
        callback_threshold (_type_): _description_
        patience (_type_): _description_
        params (_type_): _description_
        experiment_number (_type_): _description_
        save_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Initialize model.
    model = build_model(**params)

    # Create early-stopping callback.
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_root_mean_squared_error',
        min_delta=callback_threshold,
        patience=patience,
        restore_best_weights=True
    )

    # Fit model.
    history = model.fit(
        X_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[callback]
    )

    # Save training curve.
    image_path = os.path.join(ROOT_DIR, 'experiments', save_name, 'images', f'model-{experiment_number}-training-curve.jpg')
    save_training_curve(history=history, image_path=image_path)

    # Calculate, save, and return desired metrics for validation set.
    results_dictionary = evaluate_final_performance(
        model, params,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    return results_dictionary

def save_training_curve(history, image_path):
    """_summary_

    Args:
        history (_type_): _description_
        image_path (_type_): _description_
    """
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch

    fig, ax = plt.subplots(1, figsize=(8,6))
    num_epochs = model_history.shape[0]

    ax.plot(np.arange(0, num_epochs), model_history["root_mean_squared_error"],
            label="Training RMSE")
    ax.plot(np.arange(0, num_epochs), model_history["val_root_mean_squared_error"],
            label="Validation RMSE")
    ax.legend()

    plt.tight_layout()
    plt.savefig(image_path)

def evaluate_final_performance(
    model,

    params,

    X_train,
    X_val,
    X_test,

    y_train,
    y_val,
    y_test
):
    """_summary_

    Args:
        model (_type_): _description_
        params (_type_): _description_
        X_train (_type_): _description_
        X_val (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_val (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calculate, save, and return desired metrics for validation set.
    results_dict = model.evaluate(X_val, y_val, return_dict=True).copy()

    # Calculate, save, and return desired metrics for training set.
    train_results = model.evaluate(X_train, y_train, return_dict=True).copy()
    for result in train_results:
        results_dict['train_' + result] = train_results[result]

    # Add input parameters to results dict.
    results_dict.update(params)

    # Make predictions on test set.
    y_pred = model.predict(X_test)

    # Calculate statistics for training set.
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    # Append test metrics to results dict.
    results_dict['test_rmse'] = rmse_test
    results_dict['test_mae'] = mae_test
    results_dict['test_r2'] = r2_test

    return results_dict
