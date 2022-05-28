"""Experiment driver.
Creates and runs experiments and saves results.
"""
import utilities.experiment_runner as experiment_runner
from utilities.model_builder import build_model
from utilities.transformer import transformer as custom_transformer

# Set model gridsearch parameters.
model_params = {}
label_name = 'label'

# Set experiment parameters.
model_builder = build_model
transformer = custom_transformer
save_name = 'experiment-1'

sample_frac = 1
batch_size = 32
max_epochs = 100
callback_threshold = 0.01
patience = 15
sleep_time = 0

# Create experiment.
experiment = experiment_runner.TensorflowExperiment(
    save_name=save_name,
    model_builder=model_builder,
    transformer=transformer,

    create_experiment_data=False, # Only set to true on initial run.
    data_path=None
)

# Run experiment.
experiment.run_experiment(
    model_params=model_params,
    label_name=label_name,

    sample_frac=sample_frac,
    batch_size=batch_size,
    max_epochs=max_epochs,
    callback_threshold=callback_threshold,
    patience=patience,
    sleep_time=sleep_time
)
