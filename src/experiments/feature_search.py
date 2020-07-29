import os
import random
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf

from src.features import price, trendscore_features, sentiment_features, trading_features_without_price, \
    all_features_with_price, all_features_with_change, powerset, trading_features_with_price
from src.models.bidir import BidirLSTM
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, plot_one, predict_plots, write_to_json_file

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_rows', 1000)
# epsilon=1e-04


def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


if len( sys.argv ) > 1:
    folder = sys.argv[1]
else:
    folder = 'results/'
experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
experiment_results_directory = f'{folder}{os.path.basename(__file__)}/{experiment_timestamp}'


def experiment_hyperparameter_search(seed, layer_sizes, dropout_rate, loss_function, epochs, y_features, feature_list,
                                     model_generator):
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    train_portion, validation_portion, test_portion = .8, .0, .1
    X_train, y_train, X_val, y_val, X_stocks, scaler_y = load_data(feature_list, y_features, train_portion,
                                                                   test_portion,
                                                                   True)

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)
    meta = {
        'dropout': dropout_rate,
        'epochs': epochs,
        'time': sub_experiment_timestamp,
        'features': ', '.join(feature_list),
        'model-type': model_generator.__name__,
        'layer-sizes': f"[{', '.join(str(x) for x in layer_sizes)}]",
        'loss': loss_function,
        'seed': seed,
        'X-train-shape': list(X_train.shape),
        'X-val-shape': list(X_val.shape),
        'y-train-shape': list(y_train.shape),
        'y-val-shape': list(y_val.shape),
        'X-stocks': list(X_stocks)
    }

    model = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                            dropout=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size, epochs=epochs, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=1000, restore_best_weights=True)]
                        )
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('is bidir?')
    print(model_generator == BidirLSTM)

    evaluation = predict_plots(model, X_train, y_train, X_val, y_val, scaler_y, y_features[0], X_stocks,
                               directory, is_bidir=model_generator == BidirLSTM)
    plot_one('Loss history', [history.history['loss'], history.history['val_loss']],
             ['Training loss', 'Validation loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.svg', 10)

    write_to_json_file(history.history, f'{directory}/loss_history.json', )
    write_to_json_file(evaluation, f'{directory}/evaluation.json')
    write_to_json_file(meta, f'{directory}/meta.json', )


configurations = [
    {
        'lstm_type': StackedLSTM,
        'layers': [160]
    },
]

n = 10000
number_of_epochs = 100

feature_subsets = [['change', 'open', 'high', 'low', 'price'], ['change', 'direction', 'change'], ['change', 'volume']]
 
print(feature_subsets)
for seed in range(3)[:n]:
    for features in feature_subsets[:n]:
        for configuration in configurations:
            experiment_hyperparameter_search(seed=seed, layer_sizes=configuration['layers'],
                                             dropout_rate=.0,
                                             loss_function='mae',
                                             epochs=number_of_epochs,
                                             y_features=['next_change'],
                                             feature_list=list(OrderedDict.fromkeys(features)),
                                             model_generator=configuration['lstm_type'])
