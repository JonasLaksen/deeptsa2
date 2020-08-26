import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from src.pretty_print import pretty_print_evaluate
from src.utils import load_data, evaluate, transform_from_change_to_price

feature_list = ['price', 'high', 'low', 'open', 'volume', 'direction',
                'neutral_prop', 'positive_prop', 'negative_prop', 'negative', 'positive', 'neutral',
                'trendscore']


def naive_model(reference_feature):
    if (reference_feature == None):
        _, (naive_train, naive_val, naive_test), *_ = load_data([], ['price'], False)
        return (np.zeros(naive_train.shape), np.zeros(naive_val.shape), np.zeros(naive_test.shape))

    _, (naive_train, naive_val, naive_test), *_ = load_data([], [reference_feature], False)
    return (naive_train, naive_val, naive_test)


def svm(X_train, X_test, y_train, y_test):
    result = []

    for i in range(X_train.shape[0]):
        svm_model = SVR()
        svm_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = svm_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    result = []

    for i in range(X_train.shape[0]):
        linear_model = LinearRegression()
        linear_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = linear_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def ridge_regression(X_train, X_test, y_train, y_test):
    result = []

    for i in range(X_train.shape[0]):
        logistic_model = Ridge()
        y = y_train[i].reshape(-1)
        logistic_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = logistic_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def gaussian_process(X_train, X_test, y_train, y_test):
    result = []

    for i in range(X_train.shape[0]):
        gaussian_model = GaussianProcessRegressor()
        y = y_train[i].reshape(-1)
        gaussian_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = gaussian_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def main(y_type, on_test_set=False):
    (X_train, X_val, X_test), (y_train, y_val, y_test), _, scaler_y = load_data(feature_list, y_type, False)

    (_, result_val, result_test) = naive_model((y_train, y_val, y_test), y_type[0])
    # da = direction_accuracy(y_val, price_val, price_val, y_type[0])
    # print(da)
    # return
    # result= naive_model(y_train, y_test, scaler_y, y_type[0])
    # result, y = linear_regression(X_train, X_test, y_train[:,:,0], y_test[:,:,0])
    # result, y = ridge_regression(X_train, X_test, y_train, y_test)

    # Not in use
    # result, y = gaussian_process(X_train, X_val, y_train, y_val)
    # result, y = svm(X_train, X_test, y_train, y_test)

    # result_val = scaler_y.inverse_transform(result_val)
    # result_test = scaler_y.inverse_transform(result_test)

    # y_val = scaler_y.inverse_transform(y_val)  # [:,:,np.newaxis])
    # y_test = scaler_y.inverse_transform(y_test)  # [:,:,np.newaxis])

    # plot("Baseline model", stock_list, result, y)
    if (y_type[0] == 'next_change'):
        (_, result_val, result_test), (_, y_val, y_test) = transform_from_change_to_price(np.zeros(X_train.shape[:2]),
                                                                                          result_val, result_test)
        y_type = ['next_price']

    _, (price_train, price_val, price_test), *_ = load_data([], ['price'], False)
    evaluation_val = evaluate(result_val[:, :], y_val[:, :], price_val[:, :, 0], y_type=y_type[0])
    evaluation_test = evaluate(result_test[:, :], y_test[:, :], price_test[:, :, 0], y_type=y_type[0])

    print(f'Val: {evaluation_val}')
    print(f'Test: {evaluation_test}')
    # print(f'Test: {evaluation_test}')
    return (result_val, result_val), (y_val, y_val)


def compare_with_model(y_types):
    result, y = main(y_types)
    with open(
            'server_results/context_feature_search.py/2020-07-10_20.29.39/StackedLSTMWithState-160-2020-07-10_20.41'
            '.37/evaluation.json') as json_file:
        json_content = json.load(json_file)
        context_search_results = json_content['validation']
    evaluation = evaluate(result.reshape((result.shape[0], -1)), y.reshape((y.shape[0], -1)), y_type=y_types[0])
    print(pretty_print_evaluate(evaluation, context_search_results))


def naive_next_price_using_next_open():
    X_train, y_train, X_test, y_test, y_dir, scaler_y = load_data(['next_open'], ['next_price'], .8, .1,
                                                                  should_scale_y=False)
    # X_train, y_train, X_test, y_test, y_dir, scaler_y = load_data(['price'], [ 'next_open' ], .8, .1,
    # should_scale_y=False)
    result = np.concatenate((X_train, X_test), axis=1)
    y = np.concatenate((y_train, y_test), axis=1)
    evaluation = (evaluate(result.reshape((result.shape[0], -1)), y.reshape((y.shape[0], -1)), y_type='next_price'))


# main(['next_change'], on_test_set=True)
# main(['next_price'], on_test_set=True)
# compare_with_model([ 'next_price' ])
# naive_next_price_using_next_open()
