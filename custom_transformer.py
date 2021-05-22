from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ShapeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, window):
        self.window = window

    # https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman
    @staticmethod
    def sliding_window_slicing(a, no_items):
        """This method performs sliding window slicing of numpy arrays

        Parameters
        ----------
        a : numpy
            An array to be slided in subarrays
        no_items : int
            Number of sliced arrays or elements in sliced arrays

        Return
        ------
        numpy
            Sliced numpy array
        """
        no_elements = no_items
        no_slices = len(a) - no_elements + 1
        if no_slices <= 0:
            raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))

        subarray_shape = a.shape[1:]
        shape_cfg = (no_slices, no_elements) + subarray_shape
        strides_cfg = (a.strides[0],) + a.strides
        as_strided = np.lib.stride_tricks.as_strided  # shorthand
        data = as_strided(a, shape=shape_cfg, strides=strides_cfg)
        return np.squeeze(data)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return self.sliding_window_slicing(X, self.window)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(self, X, y=None, **fit_params)

def test():
    x = np.arange(50).reshape(10, 5)
    transformer = ShapeTransformer(window=1)
    transformer.fit(x)
    transformer.transform(x)