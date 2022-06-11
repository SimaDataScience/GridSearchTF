"""Utility functions for saving experiment results."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_training_curve(history, image_path: str, monitor_metric: str):
    """ Create and save image from training history.

    Args:
        history : History from fit method.
        image_path (str): Desired path for output file.
        monitor_metric (str): Metric from history from which to create curves.
    """
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch

    _, ax = plt.subplots(1, figsize=(8,6))
    num_epochs = model_history.shape[0]

    ax.plot(np.arange(0, num_epochs), model_history[monitor_metric],
            label="Training Curve")
    ax.plot(np.arange(0, num_epochs), model_history['val_' + monitor_metric],
            label="Validation Curve")
    ax.legend()

    plt.tight_layout()
    plt.savefig(image_path)

def evaluate_final_performance(
    model,
    params,

    training_dataset,
    validation_dataset,
    testing_dataset
) -> dict:
    """ Evaluate model performance on train, validation, and test datasets.

    Args:
        model : Model to evaluate.
        params (dict): Parameters that were used to build model.
        training_dataset (tf.data.Dataset): Training dataset.
        validation_dataset (tf.data.Dataset): Validation dataset.
        testing_dataset (tf.data.Dataset: Test dataset.

    Returns:
        dict: Dictionary of model performance on train, validation, and test set.
    """
    results_dict = params.copy()

    # Calculate, save, and return desired metrics for test set.
    test_results = model.evaluate(testing_dataset, return_dict=True).copy()
    for result in test_results:
        results_dict['test_' + result] = test_results[result]

    # Calculate, save, and return desired metrics for training set.
    train_results = model.evaluate(training_dataset, return_dict=True).copy()
    for result in train_results:
        results_dict['train_' + result] = train_results[result]

    # Calculate, save, and return desired metrics for validation set.
    validation_results = model.evaluate(validation_dataset, return_dict=True).copy()
    for result in validation_results:
        results_dict['val_' + result] = validation_results[result]

    return results_dict
