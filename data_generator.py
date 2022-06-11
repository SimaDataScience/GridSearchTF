"""Helper functions for manipulating and loading datasets."""
import subprocess
import tensorflow as tf

def find_number_rows(file_path: str) -> int:
    """ Return the number of rows that a csv contains,
    without reading entire file into memory.

    Args:
        file_path (str): Path to csv file.

    Returns:
        int: Number of rows in csv file.
    """
    num_lines = int(subprocess.check_output(f'wc -l {file_path}', shell=True).split()[0]) - 1

    return num_lines

def make_tensorflow_dataset(
    data_path : str,
    label_name : str,
    batch_size : int,
    epochs : int,
    buffer_size : int
) -> tf.data.Dataset:
    """ Create tensorflow dataset from path to csv file.

    Args:
        data_path (str): Path to training data.
        label_name (str): Column name of label.
        batch_size (int): Batch size for data generator.
        epochs (int): Number of epochs to generate.
        buffer_size (int): Number of samples held in memory during shuffle.

    Returns:
        tf.data.Dataset: Tensorflow dataset.
    """
    dataset = tf.data.experimental.make_csv_dataset(
        data_path,
        batch_size=batch_size,
        label_name=label_name,
        num_epochs=epochs,
        shuffle_buffer_size=buffer_size
    ).map(prepare_arrays)

    return dataset

def prepare_arrays(features, labels) -> tuple:
    """Convert 'tf.data.experimental.make_csv_dataset' output to expected tensor format.

    Args:
        features : Feature output from 'make_csv_dataset'.
        labels : Label ouutput from 'make_csv_dataset'.

    Returns:
        tuple: (batch_input_array, batch_output_array)
    """
    inputs = []
    for feature in features:
        feature_array = tf.cast(features[feature], tf.float32)
        inputs.append(feature_array)

    transpose_array = tf.stack(inputs, axis=0)
    input_array = tf.transpose(transpose_array)

    return input_array, labels
