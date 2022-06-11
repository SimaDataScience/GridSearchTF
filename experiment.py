"""Module to create experiment and run gridsearch experiments."""
import os
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import tensorflow as tf

import data_generator as datagen
import experiment_utilities as utils

tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()

class GridSearchTF:
    """ Class to run gridsearch experiment.

    Attributes:
        save_path : Directory for storing experiment results.
        save_name : Experiment name.
        model_builder : Model building function for experiment.
        df_results : Pandas dataframe results container.
        train_samples : Number of samples in training set.
        validation_samples : Number of samples in validation set.
        test_samples : Number of samples in testing set.
        best_parameters : Experiment parameters that yielded the lowest loss.

    Methods:
        run_experiment : Run full experiment with given parameters.
    """
    def __init__(self, model_builder, experiment_name: str, save_path: str) -> None:
        # Create directories for results.
        experiment_folder = os.path.join(save_path, experiment_name)
        Path(experiment_folder).mkdir(parents=True, exist_ok=True)

        images_folder = os.path.join(experiment_folder, 'images')
        Path(images_folder).mkdir(parents=True, exist_ok=True)

        self.save_path = experiment_folder
        self.save_name = experiment_name
        self.model_builder = model_builder
        self.df_results = pd.DataFrame()

        self.train_samples = None
        self.validation_samples = None
        self.test_samples = None
        self.best_parameters = None

    def run_experiment(
        self,

        train_path: str,
        validation_path: str,
        test_path: str,
        label_name: str,
        model_params:dict,

        batch_size: int=32,
        max_epochs: int=100,
        monitor_metric: str='root_mean_squared_error',
        callback_threshold: float=0.001,
        patience: int=20,
        sleep_time: int=0,
        buffer_size: int=1000
    ):
        """ Run gridsearch experiment with given parameters,
        saving performance and training curve for each model.

        Args:
            train_path (str): Path to training set.
            validation_path (str): Path to validation set.
            test_path (str): Path to testing set.
            label_name (str): Column name of label.
            model_params (dict): Model parameters with which to perform gridsearch.
            batch_size (int, optional): Batch size.
                Defaults to 32.
            max_epochs (int, optional): Number of epochs if early stopping is not achieved.
                Defaults to 100.
            monitor_metric (str, optional): Metric to monitor for early stopping.
                Defaults to 'root_mean_squared_error'.
            callback_threshold (float, optional): Threshold for early stopping.
                Defaults to 0.001.
            patience (int, optional): Patience for early stopping.
                Defaults to 20.
            sleep_time (int, optional): Seconds to rest between training models.
                Defaults to 0.
            buffer_size (int, optional): Number of samples to hold in memory during shuffle.
                Defaults to 1000.
        """
        # Store number of samples in each dataset.
        self.train_samples = datagen.find_number_rows(train_path)
        self.validation_samples = datagen.find_number_rows(validation_path)
        self.test_samples = datagen.find_number_rows(test_path)

        # Create dictionaries containing every combination from input arguments.
        parameter_grid = ParameterGrid(model_params)

        # Run experiment.
        for trial_number, parameters in tqdm(enumerate(parameter_grid)):
            print(f'Current model parameters: {parameters}')

            experiment_results = build_and_evaluate_model(
                build_model=self.model_builder,
                model_params=parameters,

                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                label_name=label_name,

                save_path=self.save_path,

                batch_size=batch_size,
                max_epochs=max_epochs,
                steps_per_epoch=self.train_samples // batch_size,
                validation_steps=self.validation_samples // batch_size,
                monitor_metric=monitor_metric,
                callback_threshold=callback_threshold,
                patience=patience,
                buffer_size=buffer_size,
                experiment_number=trial_number
            )

            # Store model performance in results container.
            self.df_results = pd.concat([self.df_results, pd.DataFrame([experiment_results])])

            # Add pause before training next model.
            time.sleep(sleep_time)

        # Save results.
        file_name = os.path.join(self.save_path, 'results-' + self.save_name + '.csv')

        ordered_results = self.df_results.sort_values('val_' + monitor_metric, ascending=False)
        self.best_parameters = ordered_results.loc[0, :].to_dict()
        self.df_results.to_csv(file_name, index_label='model_number')

def build_and_evaluate_model(
    build_model,
    model_params : dict,
    train_path : str,
    validation_path : str,
    test_path : str,
    label_name : str,
    save_path : str,
    experiment_number : int,

    batch_size : int,
    max_epochs : int,
    steps_per_epoch : int,
    validation_steps : int,
    monitor_metric : str,
    callback_threshold : float,
    patience : int,
    buffer_size : int
) -> dict:
    """ Builds and fits model with given parameters.
    Saves training curve and returns model performance on training, validation, and testing sets.

    Args:
        build_model : Function that returns tensorflow model.
        model_params (dict): Parameters required by build_model.
        train_path (str): Path to training set.
        validation_path (str): Path to validation set.
        test_path (str): Path to test set.
        label_name (str): Column name for label.
        save_path (str): Save path for training curve.
        experiment_number (int): Number tracking which model is being built.
        ** training_args.

    Returns:
        dict: Dictionary conatining model parameters and performance metrics.
    """
    # Create data generators.
    training_dataset = datagen.make_tensorflow_dataset(
        train_path, label_name, batch_size, max_epochs, buffer_size
    )
    validation_dataset = datagen.make_tensorflow_dataset(
        validation_path, label_name, batch_size, max_epochs, buffer_size
    )
    testing_dataset = datagen.make_tensorflow_dataset(
        test_path, label_name, batch_size, max_epochs, buffer_size
    )

    # Initialize model.
    model = build_model(**model_params)

    # Create early-stopping callback.
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_' + monitor_metric,
        min_delta=callback_threshold,
        patience=patience,
        restore_best_weights=True
    )

    # Fit model.
    history = model.fit(
        training_dataset,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=[callback]
    )

    # Save training curve.
    image_path = os.path.join(save_path, 'images', f'model-{experiment_number}-training-curve.jpg')
    utils.save_training_curve(history=history, image_path=image_path, monitor_metric=monitor_metric)

    # Calculate, save, and return desired metrics for validation set.
    results_dictionary = utils.evaluate_final_performance(
        model, model_params, training_dataset, validation_dataset, testing_dataset
    )

    return results_dictionary
