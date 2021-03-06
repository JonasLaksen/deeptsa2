import os
import random
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf

from src.models.bidir import BidirLSTM
from src.models.stacked_lstm import StackedLSTM
from src.models.stacked_lstm_modified import StackedLSTM_Modified
from src.pretty_print import print_for_master_thesis
from src.utils import load_data, plot_one, predict_plots, write_to_json_file

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_rows', 1000)

def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
experiment_results_directory = f'results/{os.path.basename(__file__)}/{experiment_timestamp}'


def experiment_hyperparameter_search(seed, layer_sizes, dropout_rate, loss_function, epochs, y_features, feature_list,
                                     model_generator):
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    (X_train, X_val, X_test), (y_train, y_val, y_test), X_stocks, scaler_y = load_data(feature_list, y_features)


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

    assert model_generator == StackedLSTM_Modified

    model = model_generator(
        n_features=n_features, layer_sizes=layer_sizes, return_states=False,
        dropout=dropout_rate)
    model.compile(optimizer='adam', loss=loss_function)
    y_train_list = [y_train[:,:,i:i+1] for i in range(y_train.shape[2])]
    y_val_list = [y_val[:,:,i:i+1] for i in range(y_val.shape[2])]
    history = model.fit(X_train, y_train_list,
                        validation_data=([X_val, y_val_list]),
                        batch_size=batch_size, epochs=epochs, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=1000, restore_best_weights=True)]
                        )
    if not os.path.exists(directory):
        os.makedirs(directory)

    predict_plots(model,
                  (X_train, X_val, X_test),
                  (y_train, y_val, y_test),
                  scaler_y,
                  y_features[0],
                  X_stocks,
                  directory)

    plot_one('Loss history', [history.history['loss'], history.history['val_loss']], ['Training loss', 'Test loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    write_to_json_file(str( history.history ), f'{directory}/loss_history.json', )
    write_to_json_file(meta, f'{directory}/meta.json', )


price = ['price']
trading_features = ['open', 'high', 'low', 'volume', 'direction', 'change']
trading_features_with_price = ['price'] + trading_features
sentiment_features = ['positive', 'negative', 'neutral', 'positive_prop', 'negative_prop',
                      'neutral_prop']  # , ['all_positive', 'all_negative', 'all_neutral']]#, ['all_positive', 'all_negative', 'all_neutral']]
trendscore_features = ['trendscore']

# feature_subsets = [
#     trading_features_with_price,
#     sentiment_features,
#     trendscore_features,
#     trading_features_with_price + sentiment_features,
#     trading_features_with_price + trendscore_features,
#     sentiment_features + trendscore_features,
#     trading_features_with_price + sentiment_features + trendscore_features
# ]

# price = ['prev_price_0', 'prev_price_1', 'prev_price_2'] + ['price']
# trading_features = ['prev_volume_0', 'prev_volume_1', 'prev_volume_2'] + trading_features
# sentiment_features = [f'prev_{feature}_{i}' for i, feature in enumerate(['positive', 'negative','neutral'])] + sentiment_features
# trendscore_features = [f'prev_{feature}_{i}' for i, feature in enumerate(trendscore_features)] + trendscore_features

feature_subsets = [price,
                   price + trading_features,
                   price + sentiment_features,
                   price + trendscore_features,
                   price + trading_features + sentiment_features + trendscore_features
                   ]

n = 100
number_of_epochs = 100000000

for seed in range(3)[:n]:
    for features in feature_subsets[:n]:
        experiment_hyperparameter_search(seed=seed, layer_sizes=[160],
                                         dropout_rate=.0,
                                         loss_function='mae',
                                         epochs=number_of_epochs,
                                         y_features=['next_price', 'next_direction'],
                                         feature_list=features,
                                         model_generator=StackedLSTM_Modified)


