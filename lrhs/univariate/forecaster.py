"""
Define a set of classes that function like forecasters, akin
to the R forecast package.
"""
import copy
import itertools
from typing import List, Tuple, Callable
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os 
import pickle
import pandas as pd
import numpy as np
import scipy.linalg as spla
import tensorly as tl
from scipy.fftpack import rfft, irfft, dct
from tensorly.decomposition import parafac, tucker

def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)    
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)   

from .utils import mad, multifold, rmse

TENSOR_MAX_ITER = 500

class ForecasterResult:

    def __init__(
        self,
        inputs: pd.Series,
        forecaster: "SeasonalForecaster",
        in_sample_approx: pd.Series,
        forecast: pd.Series,
        in_errors: List[float],
        out_errors: List[float],
        nr_total_params: int,
    ):
        self.inputs = inputs
        self.forecaster = forecaster
        self.in_sample_approx = in_sample_approx
        self.forecast = forecast
        self.in_errors = in_errors
        self.out_errors = out_errors
        self.nr_total_params = nr_total_params


class SeasonalForecaster:

    def __init__(
        self,
        nr_params: int,
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
    ):
        """

        :param nr_params: number of parameters for the forecaster. specific definitions change
            by forecaster.
        :param folds: a tuple representing the folds of the seasonality. faster period comes first.
            i.e., (24, 7) not (7, 24)
        :param error_callbacks:
        """
        self.nr_params = nr_params
        self.folds = folds
        self.error_callbacks = error_callbacks
        self.nr_total_params = self.nr_params

    def run_forecast(self, vals: pd.Series, nr_in_cycles: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplemented()

    def __call__(self, vals: pd.Series, nr_in_cycles: int, **kwargs):
        assert nr_in_cycles > 1, "number of cycles in sample must be > 1"
        assert nr_in_cycles * np.prod(self.folds) > len(vals) / 2, (
            "provide more data in sample then out of sample"
        )

        in_sample_approx, forecast = self.run_forecast(vals, nr_in_cycles)
        nr_in_steps = int(nr_in_cycles * np.prod(self.folds))
        data_in, data_out = (
            vals.values[:nr_in_steps], vals.values[nr_in_steps:]
        )

        in_errors = [
            callback(data_in, in_sample_approx) for callback in self.error_callbacks
        ]

        out_errors = [
            callback(data_out, forecast) for callback in self.error_callbacks
        ]
        return ForecasterResult(
            inputs=vals,
            forecaster=self,
            in_sample_approx=pd.Series(in_sample_approx, index=vals.index[:nr_in_steps]),
            forecast=pd.Series(forecast, index=vals.index[nr_in_steps:]),
            in_errors=in_errors,
            out_errors=out_errors,
            nr_total_params=self.nr_total_params,
        )


class DCTForecaster(SeasonalForecaster):

    def __init__(
        self,
        nr_params: int,
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
    ):
        """
        :param nr_params: number of DCT components to forecast with.
        """
        super().__init__(nr_params, folds, error_callbacks)

    def run_forecast(self, vals: pd.Series, nr_in_cycles: int):
        nr_in_steps = int(nr_in_cycles * np.prod(self.folds))
        data_in, data_out = (
            vals.values[:nr_in_steps], vals.values[nr_in_steps:]
        )

        z = dct(data_in)  # take the DCT

        # get the frequencies with most magnitude
        top_n = np.argsort(np.abs(z))[-self.nr_params:]

        mask = np.zeros(len(z), dtype=bool)
        mask[top_n] = True

        # zero out the other frequencies
        z_masked = np.array(z)
        z_masked[~mask] = 0

        # reconstruct
        y = dct(z_masked, type=3) / len(z) / 2
        return (
            y,  # in-sample reconstruction
            y[:len(data_out)],  # forecasts
        )


class DFTForecaster(SeasonalForecaster):

    def __init__(
        self,
        nr_params: int,
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
    ):
        """
        :param nr_params: number of DFT components to forecast with.
        """
        super().__init__(nr_params, folds, error_callbacks)

    def run_forecast(self, vals: pd.Series, nr_in_cycles: int):
        nr_steps = len(vals)
        nr_in_steps = int(nr_in_cycles * np.prod(self.folds))
        data_in, data_out = (
            vals.values[:nr_in_steps], vals.values[nr_in_steps:]
        )

        z = rfft(data_in)  # take the DCT

        # get the frequencies with most magnitude
        top_n = np.argsort(np.abs(z))[-self.nr_params:]

        mask = np.zeros(len(z), dtype=bool)
        mask[top_n] = True

        # zero out the other frequencies
        z_masked = np.array(z)
        z_masked[~mask] = 0

        # reconstruct
        y = irfft(z_masked)
        return (
            y,  # in-sample reconstruction
            y[:(nr_steps - nr_in_steps)],  # forecasts
        )

class TensorForecaster(SeasonalForecaster):
    def tensor_reconstruction(self, data_in: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    def run_forecast(self, vals: pd.Series, nr_in_cycles: int) -> Tuple[np.ndarray, np.ndarray]:
        nr_in_steps = int(nr_in_cycles * np.prod(self.folds))
        nr_out_steps = len(vals) - nr_in_steps
        nr_steps_per_cycle = np.prod(self.folds)

        nr_out_cycles = int(np.ceil(nr_out_steps / nr_steps_per_cycle))

        data_in, _ = (
            vals.values[:nr_in_steps], vals.values[nr_in_steps:]
        )

        in_sample_approx = self.tensor_reconstruction(data_in)

        cycle_approx = in_sample_approx[-nr_steps_per_cycle:]
        forecast = np.tile(cycle_approx, nr_out_cycles)[:nr_out_steps]

        return in_sample_approx, forecast


class CPForecaster(TensorForecaster):
    def __init__(
        self,
        nr_params: int,
        folds: Tuple[int],
        l2_reg: float = 0.0,
        error_callbacks: Tuple[Callable] = (rmse, mad),
        alpha: float = 1.0,
        data_idx = None,
        experiment_name = None,

    ):
        """
        Parameters
        ----------
        nr_params
            rank of tensor
        alpha
            smoothing parameter for the time factor
        """
        super().__init__(nr_params, folds, error_callbacks)
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.nr_total_params = int(nr_params * np.sum(folds))
        self.data_idx = data_idx
        self.experiment_name = experiment_name

    def tensor_reconstruction(self, data_in: np.ndarray) -> np.ndarray:
        tensor = multifold(data_in, list(self.folds))
        tensor = copy.deepcopy(tensor)

        fac = parafac(tensor, rank=self.nr_params, n_iter_max=TENSOR_MAX_ITER, tol=1.0e-13, linesearch=True, l2_reg=self.l2_reg)

        if self.alpha < 1:
            time_factor = np.array(fac.factors[0])
            for i in range(time_factor.shape[0]):
                time_factor[i] = time_factor[i] * self.alpha + time_factor[i-1] * (1 - self.alpha)

            fac.factors = [time_factor] + fac.factors[1:]

        return np.array(tl.cp_to_tensor(fac).ravel())


class TuckerForecaster(TensorForecaster):
    def __init__(
        self,
        nr_params: int,
        folds: Tuple[int],
        #l2_reg: float = 0.0,
        error_callbacks: Tuple[Callable] = (rmse, mad),
        alpha: float = 1.0,
        experiment_name = None,
        data_idx = None,
        time_R = 0
    ):
        """
        Parameters
        ----------
        nr_params
            rank of tensor
        alpha
            smoothing parameter for the time factor
        """
        super().__init__(nr_params, folds, error_callbacks)
        self.time_R = time_R
        ranks_ = self._get_tucker_ranks()
        self.alpha = alpha
        #self.l2_reg = l2_reg
        self.nr_total_params = int(
            np.sum(np.array(self.folds) * np.array(ranks_[1:])) + np.prod(ranks_[1:])
        )
        self.data_idx = data_idx
        self.experiment_name = experiment_name

    def _get_tucker_ranks(self):
        ranks = np.minimum(
            [self.nr_params for _ in range(len(self.folds))], [self.nr_params for _ in range(len(self.folds))]
        ) # HACK
        time_R = self.time_R if self.time_R > 0 else ranks[0]
        return np.r_[time_R, ranks].astype(int).tolist()

    def tensor_reconstruction(self, data_in: np.ndarray) -> np.ndarray:
        tensor = multifold(data_in, list(self.folds))
        tensor = copy.deepcopy(tensor)

        assert isinstance(self.nr_params, int)

        ranks = self._get_tucker_ranks()

        core, factors = tucker(tensor, rank=ranks, n_iter_max=TENSOR_MAX_ITER, tol=1.0e-13)        

        if self.alpha < 1:
            time_factor = np.array(factors[0])
            for i in range(time_factor.shape[0]):
                time_factor[i] = time_factor[i] * self.alpha + time_factor[i-1] * (1 - self.alpha)

            factors = [time_factor] + factors[1:]

        return np.array(tl.tucker_to_tensor((core, factors)).ravel())


class SmoothingCPForecaster(CPForecaster):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(alpha=alpha, **kwargs)


class SmoothingTuckerForecaster(TuckerForecaster):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(alpha=alpha, **kwargs)


class HoltWintersForecaster(TensorForecaster):
    def __init__(
        self,
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
        nr_params: int = 1,
        alpha: float = 0.5,
    ):
        """
        """
        super().__init__(nr_params, folds, error_callbacks)
        self.alpha = alpha
        self.nr_total_params = int(np.prod(folds))

    def tensor_reconstruction(self, data_in: np.ndarray) -> np.ndarray:
        tensor = multifold(data_in, list(self.folds))
        tensor = copy.deepcopy(tensor)

        for i in range(1, tensor.shape[0]):
            tensor[i] = tensor[i] * self.alpha + tensor[i-1] * (1 - self.alpha)

        return tensor.ravel()


class FourierBasisRegressionForecaster(SeasonalForecaster):
    def __init__(
        self,
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
        nr_params: int = 1,
    ):
        """
        Parameters
        ----------
        nr_params:
            number of Fourier basis components to generate for each fold
        """
        super().__init__(nr_params, folds, error_callbacks)
        self.nr_total_params = int(len(folds) * nr_params * 2 + 1)

    def run_forecast(self, vals: pd.Series, nr_in_cycles: int) -> Tuple[np.ndarray, np.ndarray]:
        nr_in_steps = int(nr_in_cycles * np.prod(self.folds))

        t = np.arange(len(vals), dtype=float)
        periods = [np.prod(self.folds[:n]) for n in range(1, len(self.folds) + 1)]

        basis_tuples = [
            (np.sin(2 * np.pi * t * n / P), np.cos(2 * np.pi * t * n / P)) for n, P in
            itertools.product(range(1, self.nr_params + 1), periods)
        ]

        basis = np.concatenate([np.array(x) for x in basis_tuples])
        X = np.concatenate([np.ones(basis.shape[-1])[np.newaxis, :], basis]).T
        y = vals.values

        y_in = y[:nr_in_steps]
        X_in, X_out = X[:nr_in_steps], X[nr_in_steps:]

        beta, _, _, _ = spla.lstsq(X_in, y_in)

        return X_in.dot(beta), X_out.dot(beta)
