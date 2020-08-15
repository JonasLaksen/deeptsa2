import copy
import csv
import json
from base64 import b64encode, b64decode
from zlib import compress, decompress

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.scaler import Scaler


def expand(x): return np.expand_dims(x, axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + .00000001))) * 100


def from_change_to_prices(prices, change):
    return np.add(prices, change)


def direction_accuracy(y, prediction, reference):
    one_if_positive_else_negative_one = np.vectorize(lambda element: 1 if element >= 0 else -1)

    y_change = one_if_positive_else_negative_one(y - reference)
    prediction_change = one_if_positive_else_negative_one(prediction - reference)

    same_directions = y_change * prediction_change >= 0
    individual_direction_accuracies = (np.sum(same_directions, axis=1)) / (same_directions.shape[1])
    da = np.mean(individual_direction_accuracies)

    return da


def evaluate(result, y, reference, individual_stocks=True):
    result = result.reshape(result.shape[:2])
    y = y.reshape(y.shape[:2])
    reference = reference.reshape(reference.shape[:2])

    mape = mean_absolute_percentage_error(y, result)
    mae = mean_absolute_error(y, result)
    mse = mean_squared_error(y, result)

    accuracy_direction = direction_accuracy(y[:, 1:], result[:, 1:], reference[:, 1:])

    total_evaluation = {'stock': 'All', 'MAPE': mape, 'MAE': mae, 'MSE': mse, 'DA': accuracy_direction}
    if individual_stocks:
        evaluations = []
        for i in range(result.shape[0]):
            y_ind = y[i:i + 1, :]
            result_ind = result[i:i + 1, :]
            reference_ind = reference[i:i + 1, :]
            mape = mean_absolute_percentage_error(y_ind, result_ind)
            mae = mean_absolute_error(y_ind, result_ind)
            mse = mean_squared_error(y_ind, result_ind)
            accuracy_direction = direction_accuracy(y_ind[:, 1:], result_ind[:, 1:], reference_ind[:, 1:])
            evaluation = {'stock': i, 'MAPE': mape, 'MAE': mae, 'MSE': mse, 'DA': accuracy_direction}
            evaluations.append(evaluation)
        return [total_evaluation] + evaluations
    return total_evaluation


def plot(directory, title, stocklist, graphs, legends=['Predicted', 'True value'], axises=['Day', 'Price $'],
         start_at=0):
    [plot_one(f'{title}: {stocklist[i]}', [graph[i] for graph in graphs], legends, axises,
              f'{directory}/{title}-{i}-{start_at}-{graphs[0].shape[1]}.svg', start_at=start_at) for i in
     range(len(graphs[0]))]


def plot_one(title, xs, legends, axises, filename='', start_at=0):
    assert len(xs) == len(legends)
    pyplot.title(title)
    [pyplot.plot(range(start_at, len(x)), x[start_at:],
                 label=legends[i]) for i, x in enumerate(xs)]
    pyplot.legend(loc='upper left')
    pyplot.xlabel(axises[0])
    pyplot.ylabel(axises[1])

    pyplot.grid(linestyle='--')
    if (len(filename) > 0):
        pyplot.savefig(filename, bbox_inches='tight')
    pyplot.show()
    pyplot.close()


def group_by_stock(data):
    group_by_dict = {}
    for row in data:
        try:
            group_by_dict[row[0]].append(row[0:])
        except:
            group_by_dict[row[0]] = [row[0:]]

    group_by_dict = {k: v for k, v in group_by_dict.items() if len(v) > 1600}
    data_size = len(min(group_by_dict.values(), key=len))
    # data_size = 1661
    data = list(map(lambda x: np.array(group_by_dict[x][-data_size:]), group_by_dict.keys()))
    return np.array(data)


def get_feature_list_lags(features, lags=0):
    all_features = copy.deepcopy(features)

    for feature in features:
        for i in range(lags):
            all_features = all_features + ['prev_' + feature + '_' + str(i)]

    return all_features


def write_to_csv(filename, dict):
    try:
        with open(filename, 'w') as file:
            csv.writer(file).writerows(list(map(lambda x: [x[0]] + x[1], dict.items())))
    except:
        print(filename)


def from_args_to_filename(args):
    test = json.dumps(args)
    compressed = compress(test.encode('utf-8'), 9)
    return b64encode(compressed).decode('utf-8').replace('/', '$')


def from_filename_to_args(filename):
    decoded = b64decode(filename.replace('$', '/').split('.csv')[0])
    return decompress(decoded)


def load_data(feature_list, y_features, should_scale_y=True):
    data = pd.read_csv('dataset_v2.csv', index_col=0)
    data = data.dropna()
    # data = data[data['stock'] == 'AAPL']
    # data['all_positive'] = data.groupby('date')['positive'].sum()
    # data['all_negative'] = data.groupby('date')['negative'].sum()
    # data['all_neutral'] = data.groupby('date')['neutral'].sum()
    feature_list_element_not_in_dataset = set(feature_list) - set(data.columns.values)
    if (len(feature_list_element_not_in_dataset) > 0):
        raise Exception(f'En feature ligger ikke i datasettet {feature_list_element_not_in_dataset}')

    X = data['stock'].values.reshape(-1, 1)

    try:
        values = data[[x for x in feature_list if x is not 'trendscore']].values
        X = np.append(X, values, axis=1)
    except:
        # If there are no features to be scaled an error is thrown, e.g. when feature list only consists of trendscore
        pass

    if ('trendscore' in feature_list):
        X = np.append(X, data['trendscore'].values.reshape(-1, 1), axis=1)

    y = data[y_features].values
    # y = data[y_type].values.reshape(-1, 1)
    from_index = 0
    to_index = 15
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)
    y = group_by_stock(y)[from_index:to_index, :, ]

    X = group_by_stock(X)[from_index:to_index, :, ]
    train_size = int(X.shape[1] * .8)
    val_size = int(X.shape[1] * .1)
    test_size = int(X.shape[1] * .1)

    X_train = X[:, :train_size, 1:]
    X_val = X[:, train_size:train_size + val_size, 1:]
    X_test = X[:, train_size + val_size:, 1:]

    y_train = y[:, :train_size, 1:]
    y_val = y[:, train_size:train_size + val_size, 1:]
    y_test = y[:, train_size + val_size:, 1:]

    # X_train = np.add.reduce(X_train, 0).reshape((1, X_train.shape[1], X_train.shape[2]))
    # X_test = np.add.reduce(X_test, 0).reshape((1, X_test.shape[1], X_test.shape[2]))
    # y_train = np.add.reduce(y_train, 0).reshape((1, y_train.shape[1], y_train.shape[2]))
    # y_test = np.add.reduce(y_test, 0).reshape((1, y_test.shape[1], y_test.shape[2]))

    X_scaler = Scaler()
    y_scaler = Scaler()

    if should_scale_y:
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        X_val = X_scaler.transform(X_val)
        X_test = X_scaler.transform(X_test)

        y_scaler.fit(y_train)
        y_train = y_scaler.transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)

    if (X_train.shape[2] != len(feature_list)):
        raise Exception('Lengden er feil')

    return (X_train, X_val, X_test), (y_train, y_val, y_test), X[:, 0, 0], y_scaler


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def transform_from_change_to_price(change_train, change_val, change_test):
    _, (price_train, price_val, price_test), *_ = load_data(['prev_price_1'], ['price', 'next_price'], False)

    result_train = np.add(price_train[:, :, 0], change_train.reshape(change_train.shape[:2]))[:, :, np.newaxis]
    result_val = np.add(price_val[:, :, 0], change_val.reshape(change_val.shape[:2]))[:, :, np.newaxis]
    result_test = np.add(price_test[:, :, 0], change_test.reshape(change_test.shape[:2]))[:, :, np.newaxis]
    y_train = price_train[:, :, 1][:, :, np.newaxis]
    y_val = price_val[:, :, 1][:, :, np.newaxis]
    y_test = price_test[:, :, 1][:, :, np.newaxis]
    return (result_train, result_val, result_test), (y_train, y_val, y_test)


def predict(model, X_partitions, y_partitions, scaler_y):
    (X_train, X_val, X_test) = X_partitions
    (y_train, y_val, y_test) = y_partitions

    X = np.concatenate((X_train, X_val, X_test), axis=1)

    y = np.concatenate((y_train, y_val, y_test), axis=1)
    y_inverse = scaler_y.inverse_transform(y)

    result = model.predict([X])
    result_inverse = scaler_y.inverse_transform(result)

    result_train, result_val, result_test, *_ = np.split(result_inverse[:, :, :1],
                                                         [X_train.shape[1],
                                                          X_train.shape[1] + X_val.shape[1],
                                                          X.shape[1]], axis=1)
    y_train, y_val, y_test, *_ = np.split(y_inverse[:, :, :1],
                                          [X_train.shape[1],
                                           X_train.shape[1] + X_val.shape[1],
                                           X.shape[1]], axis=1)

    return (result_train, result_val, result_test), (y_train, y_val, y_test)


def plot_results(result_partitions, y_partitions, naive_partitions, directory, stocklist, y_type):
    (result_train, result_val, result_test) = result_partitions
    (y_train, y_val, y_test) = y_partitions
    (naive_train, naive_val, naive_test) = naive_partitions

    y_axis_label = 'Change $' if y_type == 'next_change' else 'Price $'

    plot(directory, f'Training', stocklist, [result_train, y_train], ['Predicted', 'True value'], ['Day', y_axis_label])
    plot(directory, 'Validation', stocklist, [result_val, y_val], ['Predicted', 'True value'], ['Day', y_axis_label])
    plot(directory, 'Test', stocklist, [result_test, y_test], ['Predicted', 'True testue'], ['Day', y_axis_label])

    [plot(directory, 'Validation', stocklist,
          [result_val[:, :(i + 1) * 25], naive_val[:, :(i + 1) * 25],
           y_val[:, :(i + 1) * 25]], ['LSTM', 'Naive', 'True value'], ['Day', y_axis_label], start_at=i * 25) for i in
     range(6)]
    [plot(directory, 'Test', stocklist,
          [result_test[:, :(i + 1) * 25], naive_test[:, :(i + 1) * 25],
           y_test[:, :(i + 1) * 25]], ['LSTM', 'Naive', 'True value'], ['Day', y_axis_label], start_at=i * 25) for i in
     range(6)]


def predict_plots(model,
                  X_partitions,
                  y_partitions,
                  output_feature,
                  reference_feature,
                  scaler_y,
                  stocklist,
                  directory):
    (result_train, result_val, result_test), (y_train, y_val, y_test) = predict(model,
                                                                                X_partitions,
                                                                                y_partitions,
                                                                                scaler_y)
    is_change = 'change' in output_feature

    from src.baseline_models import naive_model
    (naive_train, naive_val, naive_test) = naive_model(reference_feature if not is_change else None)

    plot_results((result_train, result_val, result_test),
                 (y_train, y_val, y_test),
                 (naive_train, naive_val, naive_test),
                 directory,
                 stocklist,
                 output_feature)

    if (is_change):
        (result_train, result_val, result_test), \
        (y_train, y_val, y_test) = transform_from_change_to_price(result_train,
                                                                  result_val,
                                                                  result_test)
        _, (naive_train, naive_val, naive_test), *_ = naive_model([reference_feature])

    train_evaluation = evaluate(result_train, y_train, naive_train)
    val_evaluation = evaluate(result_val, y_val, naive_val)
    test_evaluation = evaluate(result_test, y_test, naive_test)

    naive_val_evaluation = evaluate(naive_val, y_val, naive_val)
    naive_test_evaluation = evaluate(naive_test, y_test, naive_test)

    print('Training:', train_evaluation[0])
    print('Val: ', val_evaluation[0])
    print('Test: ', test_evaluation[0])

    write_to_json_file(train_evaluation, f'{directory}/train_evaluation.json')
    write_to_json_file(val_evaluation, f'{directory}/val_evaluation.json')
    write_to_json_file(test_evaluation, f'{directory}/test_evaluation.json')

    write_to_json_file(naive_val_evaluation, f'{directory}/naive_val_evaluation.json')
    write_to_json_file(naive_test_evaluation, f'{directory}/naive_test_evaluation.json')


def write_to_json_file(dictionary, filepath):
    with open(filepath, 'a+') as f:
        f.write(json.dumps(dictionary, indent=4))
