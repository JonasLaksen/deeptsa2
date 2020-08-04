price = ['price']
change = ['change']
trading_features_without_price = ['open', 'high', 'low', 'volume', 'direction', 'change']
trading_features_without_change = ['open', 'high', 'low', 'volume', 'direction', 'price']
trading_features_with_price = ['price'] + trading_features_without_price
sentiment_features = ['positive', 'negative', 'neutral', 'positive_prop', 'negative_prop',
                      'neutral_prop']
trendscore_features = ['trendscore']

price_changes_today = [ 'open_close_change', 'high_close_change', 'low_close_change' ]

all_features_with_price = [
    price,
    price + trading_features_without_price,
    price + sentiment_features,
    price + trendscore_features,
    price + trading_features_without_price + sentiment_features + trendscore_features
]

all_features_with_change = [
    change,
    change + trading_features_without_change,
    change + sentiment_features,
    change + trendscore_features,
    change + trading_features_without_change + sentiment_features + trendscore_features
]

the_final_features = [
    price,
    change,
    price + trendscore_features + change,
    trading_features_with_price,
    change + ['positive'] + price]


def multiple_time_steps(features):
    return [f'prev_{feature}_{i}' for i in range(3) for feature in features]


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
