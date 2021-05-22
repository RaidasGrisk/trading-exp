from private import mongo_details
from pymongo import MongoClient
import pandas as pd
from typing import Union, List


def _get_reddit_ticker_stats(ticker: Union[str, None]):
    # db connection
    db_client = MongoClient(**mongo_details)

    with db_client:
        # count by date
        cur = db_client.reddit.data.aggregate([
            {  # match specific fields
                '$match': {'$and': [
                    {'data.ups': {'$gte': 10}},
                    {'data.title': {'$exists': True}},
                    {'metadata.topics.direct': {'$in': [ticker]}} if ticker else {}
                ]}

            },
            {  # group by
                '$group': {
                    # double timestamp to date:
                    # https://medium.com/idomongodb/mongodb-unix-timestamp-to-isodate-67741ab32078
                    '_id': {'$toDate': {'$multiply': ['$data.created_utc', 1000]}},
                    'total': {'$sum': 1}
                }
            }
        ])

    df = pd.DataFrame(cur).set_index('_id')
    df = df.groupby(pd.Grouper(freq='D')).sum()
    return df


def get_ticker_likes():
    pass


def get_reddit_data(tickers: List[str]):
    df = pd.DataFrame()
    df['total'] = _get_reddit_ticker_stats(None)['total']
    for ticker in tickers:
        df_ticker = _get_reddit_ticker_stats(ticker)
        df[ticker] = df_ticker['total'] / df['total']
        print('reddit', ticker)
    return df[tickers]


# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()
# model.fit(x_transformed, y_transformed['GOOGL'])
#
# importance = model.feature_importances_
# pd.DataFrame(x_transformed.columns, index=model.feature_importances_).sort_index()
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()