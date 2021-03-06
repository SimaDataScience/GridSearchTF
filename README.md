# GridSearchTF
Module to perform a small gridsearch experiment for Tensorflow models.
Trains a model with each parameter combination, then saves results and training curves to desired directory.

## Example Implementation
```
from GridSearchTF.experiment import GridSearchTF

MODEL_BUILDER = build_model # Function that accepts params and returns model.

grid_search = GridSearchTF(
    model_builder=MODEL_BUILDER,
    experiment_name='EXPERIMENT-NAME', # Name for experiment.
    save_path='.../RESULTS-DIRECTORY' # Parent directory for results folder.
)

grid_search.run_experiment(
    train_path='.../DATA-TRAIN.CSV',
    validation_path='DATA-VAL.CSV',
    test_path='DATA-TEST.csv',
    label_name='LABEL_NAME',
    model_params={'PARAM_1':[0, 1], 'PARAM_2':[0, 1]} # Parameters for build_model.

    # Optional parameters.
    # batch_size=32,
    # max_epochs=100,
    # monitor_metric='root_mean_squared_error',
    # callback_threshold=0.001,
    # patience=20,
    # sleep_time=0,
    # buffer_size=1000
)
```
