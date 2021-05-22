import tensorflow as tf


class Regression(tf.keras.Model):

    def __init__(self, output_dim, model_compile_args=None):
        super(Regression, self).__init__()

        # this complicates the object, but must initiate
        # these params together with the model if I want
        # to use tf.keras.wrappers.scikit_learn.KerasRegressor
        if not model_compile_args:
            self.model_compile_args = {
                'optimizer': 'adam',
                'loss': 'mae',
                # 'metrics': None,
                # 'loss_weights': None,
                # 'weighted_metrics': None,
                # 'run_eagerly': None
            }
        else:
            self.model_compile_args = model_compile_args

        self.w = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )

    def __call__(self, x, training=None, mask=None):
        y_ = self.w(x)
        return y_

    def build_fn(self):
        # this is needed because
        # tf.keras.wrappers.scikit_learn.KerasRegressor
        # expects tf.keras.Model.compile to return self
        # but it does not do that by default
        self.compile(**self.model_compile_args)
        return self

    def get_keras_model(self):
        train_params = {
            'batch_size': None,
            'epochs': 50,
            'verbose': 1,
            'callbacks': None,
            'validation_split': 0.,
            'validation_data': None,
            'shuffle': True,
            'class_weight': None,
            'sample_weight': None,
            'initial_epoch': 0,
            'steps_per_epoch': None,
            'validation_steps': None,
            'validation_batch_size': None,
            'validation_freq': 1,
            'max_queue_size': 10,
            'workers': 4,
            'use_multiprocessing': True
        }

        keras_model = tf.keras.wrappers.scikit_learn.KerasRegressor(
            build_fn=self.build_fn, **train_params
        )
        return keras_model


# test
def test():

    regression = Regression(1)

    x = tf.constant([[1, 1], [1, 1]])
    y = tf.constant([[1], [1]])
    regression(x)

    keras_model = regression.get_keras_model()
    keras_model.fit(x=x, y=y)


class AllocationStrategy:

    # TODO:
    # set activation to sigmoid allowing all zeros
    # then distribute to each final neuron so that
    # none zero values add up to 1 else all 0

    # kind of if value is below some threshold
    # then do not buy at all.

    def __init__(self, output_dim):
        super(AllocationStrategy, self).__init__()
        self.output_dim = output_dim
        self.w = tf.keras.layers.Dense(
            # +1 for not allocating funds at all
            output_dim + 1,
            activation='softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )

    def __call__(self, x, training=None, mask=None):
        y_ = self.w(x)
        # -1 for col of not allocating any funds
        return y_[:, :-1]

    def return_loss(self, y_, returns, penalty_alpha=0.1):
        penalty = tf.reduce_sum(tf.square(y_ - 1/self.output_dim))
        portfolio_returns = -tf.reduce_mean(tf.reduce_sum(y_ * returns, axis=1))
        return portfolio_returns + penalty * penalty_alpha

    def sharpe_loss(self, y_, returns, penalty_alpha=0.1):
        penalty = tf.reduce_sum(tf.square(y_ - 1/self.output_dim))
        portfolio_returns = -tf.reduce_mean(tf.reduce_sum(y_ * returns, axis=1))
        portfolio_std = tf.math.reduce_std(tf.reduce_sum(y_ * returns, axis=1))
        return portfolio_returns / portfolio_std + penalty * penalty_alpha

    def combination_loss(self, y_, returns, penalty_alpha=0.1):
        return self.return_loss(y_, returns, penalty_alpha) * 20 + \
               self.sharpe_loss(y_, returns, penalty_alpha)