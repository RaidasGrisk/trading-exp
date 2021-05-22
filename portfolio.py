'''
TODO: SEC fillings data https://github.com/zpetan/sec-13f-portfolio-python

'''

# https://github.com/fmilthaler/FinQuant
# https://github.com/tradytics/eiten

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pylab as plt

from typing import List
from copy import deepcopy

from reddit_data import get_reddit_data

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import plot_partial_dependence

import matplotlib
matplotlib.rcParams.update({'font.size': 6})

# options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def pandify(class_type: type):
    """Decorator for having a standard scikit-learn transformer output dataframes.

    A standard transformer is a transformer that outputs the same number of columns
    as it receives as input.

    source: https://github.com/koaning/scikit-lego/issues/304
    """

    class PandasTransformer(class_type):
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            result = super().transform(X)

            if X.shape[1] == result.shape[1]:
                cols = [c for c in X.columns]
            else:
                cols = range(0, result.shape[1])

            return pd.DataFrame(result, index=X.index, columns=cols)

    # For later object introspection, as PandasTransformer is not a helpful class type.
    # PandasTransformer.__name__ = class_type.__name__
    # PandasTransformer.__doc__ = class_type.__doc__

    return PandasTransformer


class PortfolioData:

    def __init__(self):
        self.tickers = []
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.train_mask = None
        self.test_mask = None

    @staticmethod
    def _download_ticker_data(ticker: str, start: str) -> pd.DataFrame:
        """
        :param ticker: string with ticker name
        :param start: string with date
        :return:
        """
        ticker_ = yf.Ticker(ticker)
        data = ticker_.history(start=start)
        return data

    def download_data(self, tickers: List[str], start: str = '2017-01-01') -> None:
        """

        :param tickers: list of strings with ticker names
        :param start: string with date to start downloading data from
        :return:
        """
        self.tickers = tickers
        self.x = pd.DataFrame()
        for ticker in tickers:
            ticker_data = self._download_ticker_data(ticker, start)
            ticker_data.columns = [ticker + '_' + col for col in ticker_data.columns]
            self.x = pd.concat([self.x, ticker_data], axis=1)
        # self.x = self.x.sort_index(ascending=False)

    @staticmethod
    def _get_default_indicators(data: pd.DataFrame, **args: dict) -> pd.DataFrame:
        cols = data.columns
        return ta.add_all_ta_features(
            data,
            **args,
            fillna=False
        ).drop(cols, axis=1)

    def get_reddit_stats(self) -> None:
        df = get_reddit_data(self.tickers)
        df.columns = [i + '_reddit_vol' for i in df.columns]
        self.x = pd.concat([self.x, df], axis=1)

    def add_indicators(self, tickers: List[str] = None):
        tickers = self.tickers if not tickers else self.tickers
        for ticker in tickers:
            ticker_cols = [col for col in self.x.columns if col.startswith(ticker)]
            col_map = {
                'open': ticker + '_Open',
                'high': ticker + '_High',
                'low': ticker + '_Low',
                'close': ticker + '_Close',
                'volume': ticker + '_Volume'
            }
            ticker_indi = self._get_default_indicators(self.x[ticker_cols], **col_map)
            ticker_indi.columns = [ticker + '_' + col for col in ticker_indi.columns]
            self.x = pd.concat([self.x, ticker_indi], axis=1)

        # some features contain null values only
        # this create problems down the line later on
        # for example when imputing missing values
        # so lets get rid of them here
        self.x = self.x.dropna(axis=1, how='all')

    def make_targets(self) -> None:

        # debug by checking:
        close_price_cols = [i for i in self.x.columns if 'Close' in i]
        self.y = self.x[close_price_cols].pct_change().shift(-1)

        # rename columns
        self.y.columns = self.tickers

    def make_data_split_masks(self):
        train_index, test_index = train_test_split(self.x.index, shuffle=False)
        self.train_mask = self.x.index.isin(train_index)
        self.test_mask = self.x.index.isin(test_index)


class PortfolioPrediction:

    def __init__(self):
        self.preprocess_pipe = None
        self.preprocess_pipe_y = None
        self.model_pipes = {}
        self.model_pipe = None
        self.selected_features = {}

    def preselect_features(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            k: int = 50
    ):

        x_transformed = self.preprocess_pipe.transform(x)
        y_transformed = self.preprocess_pipe_y.transform(y)

        for stock_id, ticker in enumerate(y.columns):
            # https://stats.stackexchange.com/questions/204141/difference-between-selecting-features-based-on-f-regression-and-based-on-r2
            info = mutual_info_regression(
                x_transformed[self.selected_features[ticker]],
                y_transformed[ticker],
                random_state=0
            )
            info_sorted = info[info.argsort()]
            mask = info > info_sorted[-k+1]

            self.selected_features[ticker] = x_transformed[self.selected_features[ticker]].columns[mask]

    def select_features(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            test_fraction: float = 0.2,
            estimator: BaseEstimator = Ridge(alpha=1),
            min_features_to_select: int = 1,
            step: int = 1,
            plot: bool = True,
    ) -> None:

        # feature selection data fraction
        # split to validation dataset to score
        # for each iteration of feature selection
        # iterable yielding (train, test) splits
        n_data_points = x.shape[0]
        f_select_data = int(n_data_points * test_fraction)
        start, mid, end = 0, n_data_points - f_select_data, n_data_points
        training_mask_splits = [(
            np.array(range(start, mid)),
            np.array(range(mid, end))),
        ]

        rfecv_params = {
            'estimator': estimator,
            'step': step,
            'n_jobs': 4,
            'cv': training_mask_splits,
            'scoring': make_scorer(mean_squared_error, greater_is_better=False),
            'min_features_to_select': min_features_to_select
        }

        x_transformed = self.preprocess_pipe.transform(x)
        y_transformed = self.preprocess_pipe_y.transform(y)

        for stock_id, ticker in enumerate(y.columns):

            rfecv = RFECV(**rfecv_params)

            rfecv.fit(
                x_transformed[self.selected_features[ticker]],
                y_transformed[ticker]
            )

            self.selected_features[ticker] = \
                x_transformed[self.selected_features[ticker]].columns[rfecv.support_]

            if plot:
                plt.figure(ticker)
                plt.xlabel('Number of features')
                plt.ylabel('Cross val score')
                plt.plot(
                    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
                    rfecv.grid_scores_
                )
                plt.show()
                plt.pause(1e-09)

    def fit_models(self, x: pd.DataFrame, y: pd.DataFrame):
        for ticker in y.columns:
            self.model_pipe.fit(
                self.preprocess_pipe.transform(x)[self.selected_features[ticker]],
                self.preprocess_pipe_y.transform(y)[ticker],
            )
            self.model_pipes[ticker] = deepcopy(self.model_pipe)

    def predict_models(self, x: pd.DataFrame, y: pd.DataFrame):
        y_ = pd.DataFrame(index=y.index)
        for ticker in y.columns:
            y_[ticker] = self.model_pipes[ticker].predict(
                self.preprocess_pipe.transform(x)[self.selected_features[ticker]]
            )
        return y_

    def get_metrics(self, x: pd.DataFrame, y: pd.DataFrame):
        y_ = self.predict_models(x, y)
        return ((y - y_) ** 2).mean(axis=0)


class InstrumentModel:

    def __init__(self):
        self.pre_pipe_x = None
        self.pre_pipe_y = None
        self.model_pipe = None

    def fit_model(self, x, y):
        self.model_pipe.fit(
            self.pre_pipe_x.transform(x),
            self.pre_pipe_y.transform(y)
        )

    def transform(self):
        return


def hyper_param_tune():

    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    data = PortfolioData()
    data.download_data(['NVDA'])
    data.add_indicators()
    data.get_reddit_stats()
    data.make_targets()
    data.make_data_split_masks()

    x = data.x[data.train_mask]
    y = data.y[data.train_mask]

    # when fitting gridsearchcv we want to get
    # the score of each fit on some cv dataset
    # lets score on the last 20% of training set
    n_data_points = x.shape[0]
    f_select_data = int(n_data_points * 0.2)
    start, mid, end = 0, n_data_points - f_select_data, n_data_points
    training_mask_splits = [(
        np.array(range(start, mid)),
        np.array(range(mid, end))),
    ]

    prepr_pipe = Pipeline([
        ('SimpleImputer', SimpleImputer()),
        ('QuantileTransformer', PowerTransformer()),
        ('VarianceThreshold', VarianceThreshold()),
        ('feature_selection', SelectKBest()),
        ('RFECV', RFECV(
            estimator=Ridge(),
            step=1,
            n_jobs=1,
            cv=4,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            min_features_to_select=1)
        )
    ])

    model_pipe = Pipeline([
        ('clf', Ridge())
    ])

    pipe = Pipeline([
        ('preprocess', prepr_pipe),
        ('model', model_pipe),
    ])

    pipe.fit(x, y)
    pipe.predict(x)

    # preprocessing pipes
    # made separately to include in each model
    prepr_space = {
        'preprocess__feature_selection': [SelectKBest()],
        'preprocess__feature_selection__score_func': [f_regression, mutual_info_regression],
        'preprocess__feature_selection__k': [10, 20, 30, 40, 50],
        'preprocess__RFECV__estimator__alpha': [0, 1, 2, 3],
    }

    # https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search
    search_space = [

        # model pipes
        {
            **prepr_space,
            **{
                # 'model__clf': [Ridge()],
                'model__clf__alpha': [0, 1, 2, 4, 6, 8, 10, 15, 20],
                'model__clf__fit_intercept': [True, False]
            }
        },

        # {
        #     **prepr_space,
        #     **{
        #         'model__clf': [DecisionTreeRegressor()],
        #         'model__clf__criterion': ['mse', 'mae']
        #     },
        #
        # }
    ]

    grid = GridSearchCV(pipe, search_space, cv=training_mask_splits, n_jobs=4, scoring='r2')
    grid.fit(x.values, y.values.ravel())
    print(mean_squared_error(grid.predict(data.x[data.test_mask]), data.y[data.test_mask].fillna(method='ffill')))
    print(grid.best_estimator_.steps[-2][-1].steps[-1][1].support_)

    print(grid.best_estimator_)
    [print(i) for i in grid.cv_results_]
    [print(i) for i in grid.cv_results_['params']]


    def plot():

        # https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
        df_ = pd.DataFrame(grid.cv_results_)
        unique_params = set()
        for dict_of_params in grid.cv_results_['params']:
            [unique_params.add(str(param)) for param in [*dict_of_params]]

        nrows, ncols = int(np.ceil(len(unique_params) / 3)), 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(6, 2),)
        ax = [item for sublist in ax for item in sublist]
        i = 0

        for param in unique_params:
            param_col = 'param_' + param
            param_stats = df_[['mean_test_score', 'std_test_score', param_col]].dropna()

            # convert the fu*king functions to str
            # and strip nonsense leaving function name only
            val_type = type(param_stats[param_col].values[0])
            if val_type not in [str, int, float, bool]:
                try:
                    param_stats[param_col] = param_stats[param_col].astype(str)
                    param_stats[param_col] = param_stats[param_col].apply(lambda x: str(x).split(' ')[1])
                except:
                    pass

            param_stats.groupby(param_col).mean().plot(yerr='std_test_score', linestyle='--', marker='o', label='test', ax=ax[i])
            i += 1

        fig.tight_layout()

# ------------ #


def get_all_tickers() -> pd.DataFrame:
    ticker_list_url = 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt'
    ticker_data = pd.read_csv(ticker_list_url, sep='|')
    return ticker_data


# ------------ #
def make_predictions():

    tickers = [

        'GOOGL',
        # 'AMZN', 'AAPL', 'NFLX',
        # 'MSFT', 'TSLA', 'AMC', 'GME',
        # 'NVDA', 'TECH', 'INTC', 'BABA',
        # 'PYPL', 'CSCO', 'MTCH', 'ADBE', 'DBX','QQQ', 'CRM',

        # 'ATVI', 'ALXN', 'AMGN', 'ADI', 'AEP', 'ANSS', 'AMAT', 'ASML', 'TEAM',
        # 'ADSK', 'ADP', 'BIDU', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CERN',
    ]

    data = PortfolioData()
    data.download_data(tickers)
    # ----------- #
    # from sklearn.datasets import make_regression
    # X, y = make_regression(n_samples=1000, n_targets=1, n_features=100, n_informative=10, random_state=0)
    # tickers = ['A']
    # data.tickers = tickers
    # data.x, data.y = pd.DataFrame(X), pd.DataFrame(y, columns=tickers)
    # ----------- #
    data.add_indicators()
    data.get_reddit_stats()
    data.make_targets()
    data.make_data_split_masks()

    # model data
    predictions = PortfolioPrediction()

    # preprocess x
    predictions.preprocess_pipe = Pipeline([
            ('SimpleImputer', pandify(SimpleImputer)()),
            # ('PolynomialFeatures', PolynomialFeatures(degree=1)),
            ('QuantileTransformer', pandify(PowerTransformer)()),
            # ('QuantileTransformer', pandify(QuantileTransformer)()),
            # ('pca', pandify(PCA)()),
            # ('ShapeTransformer', ShapeTransformer(window=1)),
        ])

    predictions.preprocess_pipe.fit(data.x[data.train_mask])
    predictions.preprocess_pipe.transform(data.x[data.train_mask]).tail()

    # preprocess y
    predictions.preprocess_pipe_y = Pipeline([
            ('SimpleImputer', pandify(SimpleImputer)()),
            # ('ShapeTransformer', pandify(ShapeTransformer)(window=1)),
        ])

    predictions.preprocess_pipe_y.fit(data.y[data.train_mask])
    predictions.preprocess_pipe_y.transform(data.y[data.train_mask]).tail()

    # feature selection
    # preselect features based on some indicators
    predictions.selected_features = {ticker: data.x.columns for ticker in data.y.columns}
    predictions.selected_features = {ticker: [col for col in data.x.columns if col.startswith(ticker)] for ticker in data.y.columns}
    predictions.selected_features = {ticker: [col for col in data.x[[col for col in data.x.columns if col.startswith(ticker)]].corrwith(data.y[ticker]).abs().sort_values().dropna()[-50:].index] for ticker in data.y.columns}  # [[col for col in data.x.columns if col.startswith(ticker)]]
    predictions.preselect_features(
        x=data.x[data.train_mask],
        y=data.y[data.train_mask],
        k=50
    )

    # filter out the worst features by other means
    predictions.select_features(
        x=data.x[data.train_mask],
        y=data.y[data.train_mask],
        estimator=DecisionTreeRegressor(max_depth=20, min_samples_split=10, random_state=0),
        # estimator=RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_split=5, random_state=0),
        # estimator=Ridge(0),
        min_features_to_select=1,
        step=1,
        plot=True,
        test_fraction=0.2
    )

    # model fitting
    predictions.model_pipe = Pipeline([
        # ('LinearRegression', DecisionTreeRegressor(max_depth=4, min_samples_split=10, random_state=0)),
        # ('LinearRegression', MLPRegressor(alpha=1, random_state=0)),
        ('LinearRegression', LinearRegression(fit_intercept=True))
    ])

    predictions.fit_models(
        x=data.x[data.train_mask],
        y=data.y[data.train_mask],
    )

    y_ = predictions.predict_models(
        x=data.x[data.train_mask],
        y=data.y[data.train_mask],
    )

    # -------- #
    for ticker in data.tickers:
        ax = plot_partial_dependence(
            predictions.model_pipes[ticker],
            predictions.preprocess_pipe.transform(data.x[data.train_mask])[predictions.selected_features[ticker]],
            features=predictions.selected_features[ticker],
            kind='both', subsample=50,
            n_jobs=4, grid_resolution=20, random_state=0
        )
        ax.figure_.suptitle(ticker)
        ax.figure_.subplots_adjust(hspace=0.3)
        plt.pause(1e-10)
        break

    # run this to check if good features are selected
    import seaborn as sns
    for feature in predictions.selected_features['GOOGL']:
        sns.jointplot(
            x=predictions.preprocess_pipe.transform(data.x[data.test_mask])[feature],
            y=data.y[data.test_mask]['GOOGL'], kind='reg'
        )
        plt.pause(0.0001)


    # -------- #

    # metrics
    train_score = predictions.get_metrics(data.x[data.train_mask], data.y[data.train_mask])
    test_score = predictions.get_metrics(data.x[data.test_mask], data.y[data.test_mask])
    print(pd.concat([train_score, test_score], axis=1))

    return data, predictions


def make_strategy():

    data, predictions = make_predictions()

    y_test_raw = data.y[data.test_mask]

    y_train = predictions.preprocess_pipe_y.transform(
        data.y[data.train_mask],
    )

    y_test = predictions.preprocess_pipe_y.transform(
        data.y[data.test_mask],
    )

    y_train_ = predictions.predict_models(
        x=data.x[data.train_mask],
        y=data.y[data.train_mask],
    )

    y_test_ = predictions.predict_models(
        x=data.x[data.test_mask],
        y=data.y[data.test_mask],
    )

    # error
    train_score = predictions.get_metrics(data.x[data.train_mask], data.y[data.train_mask])
    test_score = predictions.get_metrics(data.x[data.test_mask], data.y[data.test_mask])
    print(pd.concat([train_score, test_score], axis=1))

    # plot percentages
    for ticker in y_test_.columns:
        plt.figure(ticker)
        plt.plot(y_test_[ticker], c='orange', linewidth=3.0)
        plt.plot(y_test_raw[ticker])

    # plot vs buy and hold
    for ticker in y_test_.columns:
        plt.figure(ticker)
        strat_mask = y_test_[ticker] >= 0.002
        plt.plot((y_test_raw[ticker] * strat_mask).cumsum(), c='orange', linewidth=3.0)
        plt.plot(y_test_raw[ticker].cumsum())

    # generate baseline strategy
    import numpy as np

    # get weights
    w = np.random.randint(0, 100, size=y_test.shape)
    w = w / np.atleast_2d(w.sum(axis=1)).T

    # get portfolio value
    (y_test * w).sum(axis=1).cumsum().plot()

    # generate model strategy
    from tf_model import AllocationStrategy
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    alloc_model = AllocationStrategy(output_dim=y_test.shape[1])
    alloc_model.train_loss, alloc_model.test_loss = [], []
    div_pen = 0.0001

    for _ in range(1000):
        with tf.GradientTape() as tape:
            loss_train = alloc_model.combination_loss(alloc_model(y_train_.values), y_train.values, div_pen)
            loss_test = alloc_model.combination_loss(alloc_model(y_test_.values), y_test.values, div_pen)

            grads = tape.gradient(loss_train, alloc_model.w.trainable_variables)
            optimizer.apply_gradients(zip(grads, alloc_model.w.trainable_variables))

            # print(loss.numpy())
            alloc_model.train_loss.append(loss_train)
            alloc_model.test_loss.append(loss_test)

        if _ % 200 == 0:
            plt.plot(alloc_model.train_loss, c='black')
            plt.plot(alloc_model.test_loss, c='orange')
            plt.show()
            plt.pause(1e-09)

    # plot strategy
    (y_train * alloc_model(y_train_.values).numpy()).sum(axis=1).cumsum().plot()
    (y_test * alloc_model(y_test_.values).numpy()).sum(axis=1).cumsum().plot()

    # plot weights
    pd.DataFrame(alloc_model(y_train_.values).numpy(), columns=y_train_.columns, index=y_train_.index).plot.area()
    pd.DataFrame(alloc_model(y_test_.values).numpy(), columns=y_test_.columns, index=y_test_.index).plot.area()

    pd.DataFrame(alloc_model(y_test_.values).numpy(), columns=y_test_.columns).mean(axis=0).sort_values()
    pd.DataFrame(alloc_model(y_train_.values).numpy(), columns=y_train_.columns).mean(axis=0).sort_values()