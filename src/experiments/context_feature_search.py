import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.bidir_state import BidirLSTMWithState
from src.models.spec_network import SpecializedNetwork
from src.models.stacked_lstm_state import StackedLSTMWithState
from src.utils import load_data, plot_one, predict_plots, write_to_json_file
from src.features import all_features_with_change, trading_features_without_change, sentiment_features

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_rows', 1000)




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
    print(layer_sizes)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, X_stocks, scaler_y = load_data(feature_list, y_features, train_portion,
                                                                   test_portion,
                                                                   True)
    X = np.append(X_train, X_val, axis=1)
    y = np.append(y_train, y_val, axis=1)
    stock_list = np.arange(len(X)).reshape((len(X), 1, 1))

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
    decoder = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                              dropout=dropout_rate)

    spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(stock_list), layer_sizes=layer_sizes,
                                    decoder=decoder, n_states_per_layer=2)
    spec_model.compile(optimizer=tf.optimizers.Adam(), loss=loss_function)

    history = spec_model.fit([X_train, stock_list], y_train,
                             validation_data=([X_val, stock_list], y_val),
                             batch_size=batch_size, epochs=epochs, shuffle=False,
                             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                         patience=1000, restore_best_weights=True)]
                             )

    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(spec_model, X_train, y_train, X_val, y_val, scaler_y, y_features[0], X_stocks,
                               directory, [stock_list])
    plot_one('Loss history', [history.history['loss'], history.history['val_loss']], ['Training loss', 'Test loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.svg', 10)

    write_to_json_file(history.history, f'{directory}/loss_history.json', )
    write_to_json_file(evaluation, f'{directory}/evaluation.json')
    write_to_json_file(meta, f'{directory}/meta.json', )


configurations = [
    {
        'lstm_type': StackedLSTMWithState,
        'layers': [160]
    }
]

n = 10000
number_of_epochs = 1000000

feature_subsets = [['price'] + sentiment_features, ['price'], ['price','trendscore']]


for seed in range(10)[:n]:
    for features in feature_subsets:
        for configuration in configurations:
            experiment_hyperparameter_search(seed=seed, layer_sizes=configuration['layers'],
                                             dropout_rate=0,
                                             loss_function='mae',
                                             epochs=number_of_epochs,
                                             y_features=['next_price'],
                                             feature_list=features,
                                             model_generator=configuration['lstm_type'])

# print_folder = f'results/context_feature_search.py/2020-07-16_22.45.19//*/'
# print_for_master_thesis(print_folder, ['features', 'layer'])
# print_for_master_thesis_compact(print_folder, ['features', 'layer'])
