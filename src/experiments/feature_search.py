import os
import sys
from collections import OrderedDict
from datetime import datetime

import pandas
import tensorflow as tf

from src.features import all_features_with_price
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, plot_one, write_to_json_file, predict_plots

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_rows', 1000)

if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = 'results/'
experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
experiment_results_directory = f'{folder}{os.path.basename(__file__)}/{experiment_timestamp}'


def experiment_hyperparameter_search(
        input_features,
        output_feature,
        reference_feature,
        layer_sizes,
        dropout_rate,
        loss_function,
        epochs,
        model_generator):

    print(input_features)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-' \
                f'{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    (X_train, X_val, X_test), (y_train, y_val, y_test), X_stocks, scaler_y = load_data(input_features, [output_feature])

    n_features, batch_size = X_train.shape[2], X_train.shape[0]
    meta = {
        'dropout': dropout_rate,
        'epochs': epochs,
        'time': sub_experiment_timestamp,
        'features': ', '.join(input_features),
        'model-type': model_generator.__name__,
        'layer-sizes': f"[{', '.join(str(x) for x in layer_sizes)}]",
        'loss': loss_function,
        'X-train-shape': list(X_train.shape),
        'X-val-shape': list(X_val.shape),
        'y-train-shape': list(y_train.shape),
        'y-val-shape': list(y_val.shape),
        'X-stocks': list(X_stocks)
    }

    model = model_generator(n_features=n_features,
                            layer_sizes=layer_sizes,
                            return_states=False,
                            dropout=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=1000, restore_best_weights=True)]
                        )
    if not os.path.exists(directory):
        os.makedirs(directory)

    predict_plots(model,
                  (X_train, X_val, X_test),
                  (y_train, y_val, y_test),
                  output_feature,
                  reference_feature,
                  scaler_y,
                  X_stocks,
                  directory)

    plot_one('Loss history',
             [history.history['loss'],
              history.history['val_loss']],
             ['Training loss', 'Validation loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.svg',
             10)

    write_to_json_file(history.history, f'{directory}/loss_history.json', )
    write_to_json_file(meta, f'{directory}/meta.json', )


n = 100000
number_of_epochs = 1000000

feature_subsets = all_features_with_price

print(feature_subsets)
for seed in range(3)[:n]:
    for features in feature_subsets[:n]:
        experiment_hyperparameter_search(layer_sizes=[160],
                                         dropout_rate=.0,
                                         loss_function='mae',
                                         epochs=number_of_epochs,
                                         output_feature='next_open',
                                         reference_feature='price',
                                         input_features=list(OrderedDict.fromkeys(features)),
                                         model_generator=StackedLSTM)
