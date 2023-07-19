import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorly as tl
import itertools
import string
from numpy import einsum
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from scipy import interpolate
from scipy.stats import linregress
from scipy.fftpack import rfft, irfft, dct, idct
from tensorly.decomposition import parafac, tucker


# introduce some utility functions
def repeat(ts: np.ndarray, times: int) -> np.ndarray:
    assert ts.ndim == 1
    out = np.array(ts)
    for _ in range(times - 1):
        out = np.r_[out, ts]
    return out


def fold(X: np.ndarray, n_p: int):
    """fold first mode into n_p tubes"""
    newshape = [-1, n_p] + list(X.shape[1:])
    return np.reshape(X, newshape)


def multifold(X: np.ndarray, n_ps: List[int]):
    for n_p in n_ps:
        X = fold(X, n_p)
    return X


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(
        np.nanmean(np.square(y_true - y_pred))
    )


def mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.nanmean(np.abs(y_true - y_pred)).sum()


def dct_dft_errors(data, max_params, param_sweep_scale="linear", no_params=1000, error_fn=rmse):
    # RMSEs with DCT
    param_sweep = get_param_sweep(max_params, param_sweep_scale, no_params)
    dct_errors = [
        error_fn(data, dct_reconstruct(data, p))
        for p in param_sweep
    ]

    # RMSEs with DFT
    dft_errors = [
        error_fn(data, dft_reconstruct(data, p))
        for p in param_sweep
    ]
    return dct_errors, dft_errors, param_sweep


def dct_reconstruct(data: np.ndarray, n: int):
    z = dct(data)  # take the DCT

    # get the frequencies with most magnitude
    top_n = np.argsort(np.abs(z))[-n:]

    mask = np.zeros(len(z), dtype=bool)
    mask[top_n] = True

    # zero out the other frequencies
    z_masked = np.array(z)
    z_masked[~mask] = 0

    # reconstruct
    return dct(z_masked, type=3) / len(z) / 2


def dft_reconstruct(data: np.ndarray, n: int):
    z = rfft(data)  # take the DCT

    # get the frequencies with most magnitude
    top_n = np.argsort(np.abs(z))[-n:]

    mask = np.zeros(len(z), dtype=bool)
    mask[top_n] = True

    # zero out the other frequencies
    z_masked = np.array(z)
    z_masked[~mask] = 0

    # reconstruct
    return irfft(z_masked)


def tensor_errors(data, folds, ranks, decomposition_type="parafac", error_fn=rmse):
    # with tensors
    tensor_results = []
    for rank in ranks:
        _ = print(rank) if rank % 3 == 0 else None
        data_approx, npars = tensor_reconstruction(data, folds, rank, decomposition_type=decomposition_type)
        tensor_results.append(
            [error_fn(data, data_approx), npars]
        )
    ten_errors, ten_params = zip(*tensor_results)
    return ten_errors, ten_params


# melih utility functions
def tensor_errors_test(data, test_data, folds, ranks, decomposition_type="parafac"):
    # with tensors
    tensor_results = []
    for rank in ranks:
        _ = print(rank) if (rank + 1) % 2 == 0 else None
        data_approx, npars = tensor_reconstruction(data, folds, rank, decomposition_type=decomposition_type)
        # calculate the training RMSE (we will change data approx below)
        rmse_train = rmse(data, data_approx)
        # take means of the tensor in the trivial direction
        # mean_trivial_direction = data_approx.mean(0)[np.newaxis, ...]
        # broadcast the mean to each slice in the trivial direction
        # for i in range(data_approx.shape[0]):
        #    data_approx[i, ...] = mean_trivial_direction
        tensor_results.append(
            [rmse_train, rmse(test_data, data_approx), npars]
        )
    ten_errors, test_errors, ten_params = zip(*tensor_results)
    return ten_errors, test_errors, ten_params


def get_param_sweep(max_params, param_sweep_scale, no_params):
    if param_sweep_scale == "linear":
        return np.floor(np.linspace(1, max_params, no_params)).astype(int)
    elif param_sweep_scale == "log":
        return np.unique(np.floor(np.logspace(0, np.log10(max_params), no_params))).astype(int)
    else:
        raise Exception("Param sweep scale not defined")

def create_hp_grid(hp_dict):
    keys, values = zip(*hp_dict.items())
    hp_combinations = itertools.product(*values)
    return [dict(zip(keys, hp_combination)) for hp_combination in hp_combinations]

def dct_dft_errors_test(data, test_data, max_params, param_sweep_scale, no_params):
    dct_errors, dft_errors, param_sweep = dct_dft_errors(data=data, max_params=max_params,
                                                         param_sweep_scale=param_sweep_scale, no_params=no_params)
    dct_test_errors = [
        rmse(test_data, dct_reconstruct(data, p))
        for p in param_sweep
    ]
    dft_test_errors = [
        rmse(test_data, dft_reconstruct(data, p))
        for p in param_sweep
    ]
    return dct_errors, dct_test_errors, dft_errors, dft_test_errors, param_sweep


def plot_comparison(dct_errors, dft_errors, ten_params, ten_errors, param_sweep, folds, td_params=None, td_errors=None):
    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_sweep, dct_errors, 'b.-', label="DCT")
    ax.plot(param_sweep, dft_errors, 'g.-', label="DFT")
    ax.plot(ten_params, ten_errors, 'r.-', label="CP")
    if td_params is not None:
        ax.plot(td_params, td_errors, 'm.-', label="Tucker")
    ax.axvline(np.product(folds), color='grey', linestyle='--', label='$\dim \, \mathbf{s}$')
    ax.set(xlabel="# Parameters (logarithmic)", ylabel="RMSE")
    ax.legend()
    ax.semilogx();


def get_plot_data(idx, train_datas, test_datas, freq, plot=True):
    data = pd.concat((to_pandas(train_datas[idx]), to_pandas(test_datas[idx])))
    data.index = pd.date_range(start=data.index[0], freq=freq, periods=len(data))
    if plot:
        data.plot();
    return data


def get_gluonts_dataset(dataset_name):
    dataset = get_dataset(dataset_name, regenerate=False)
    train_datas = list(iter(dataset.train))
    test_datas = list(iter(dataset.test))
    lens = [len(d["target"]) for d in train_datas]
    freqs = [d["start"].freqstr for d in train_datas]
    print(pd.Series(lens).value_counts())
    print(pd.Series(freqs).value_counts())
    del dataset
    return train_datas, test_datas, lens, freqs


def trend_cycle_decompose(df: pd.Series, w: int, df_train=None):
    assert type(df) == pd.core.series.Series
    assert type(w) == int
    assert w > 1

    dfi = df.interpolate("linear")
    trend_cycle = dfi.rolling(w).mean().fillna(method="bfill")
    residual = dfi - trend_cycle

    return trend_cycle, residual


def naive_seasonal_decompose(df: pd.Series, w: int, df_train=None):
    a = np.array(df)
    new_len_a = (len(a) // w) * w

    seasonal = multifold(a[:new_len_a], [w]).mean(0)
    seas_effect = pd.Series(
        repeat(seasonal, len(a) // w + 1)[:len(a)], index=df.index
    )

    return seas_effect, df - seas_effect


def analyze_and_plot(df, period: int, plot=True):
    tc, res_tc = trend_cycle_decompose(df, period * 2)
    seas, res_seas = naive_seasonal_decompose(res_tc, period * 7)

    r2 = np.square(seas).sum() / np.square(res_tc).sum()

    if plot:
        f, axes = plt.subplots(3, figsize=(8, 5), sharex=True)
        for ax_, title, obj in zip(
                axes,
                ["Trend-cycle", "Seasonal", "Residual"],
                [tc, seas, res_seas]
        ):
            ax_.plot(obj)
            ax_.set(title=title)

        f.suptitle(f"R^2: {r2: .2f}")
        plt.show()

    return r2

# DEPRECATED
def tc_decompose(df, w, df_train=None):
    assert type(df) == pd.core.series.Series
    assert type(w) == int
    assert w > 1
    if w / len(df) > .10:
        print("Too many null values, using linear first order polynomial for detrending.")
        ma = df.copy()
        if df_train is None:
            m = linregress(range(len(df.values)), df.values)
            ma[:] = m.intercept + m.slope * np.arange(len(df.values))
        else:
            print("Using training data for linear regression, assuming continuity.")
            m = linregress(range(len(df_train.values)), df_train.values)
            ma[:] = m.intercept + m.slope * (len(df_train) + np.arange(len(df.values)))
    else:
        if w % 2 == 0:
            lower_cumsum = df.cumsum().shift((w // 2))
            lower_cumsum.iloc[w // 2 - 1] = 0.
            ma_w = (df.cumsum().shift(-(w // 2)) - lower_cumsum) / w
            lower_cumsum = ma_w.cumsum().shift(2)
            lower_cumsum.iloc[w // 2] = 0.
            ma = (ma_w.cumsum() - lower_cumsum) / 2
        elif w % 2 == 1:
            lower_cumsum = df.cumsum().shift((w // 2 + 1))
            lower_cumsum.iloc[w // 2] = 0.
            ma = (df.cumsum().shift(-(w // 2)) - lower_cumsum) / w
        f = interpolate.interp1d(ma.reset_index(drop=True).dropna().index, ma.dropna().values, fill_value='extrapolate')
        ma[:] = f(range(len(ma)))
    return ma


def remove_ma(data, w, df_train=None):
    return data - tc_decompose(data, w, df_train=df_train)


def plot_tc_decomposition(data, ma_folds, df_train=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ma = tc_decompose(data, int(np.prod(ma_folds)), df_train=df_train)
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 4))
        data.plot(ax=axes[0])
        ma.plot(ax=axes[1])
        (data - ma).plot(ax=axes[2])


def extend_test_data(data, test_data, freq):
    temp = pd.Series(
        index=pd.date_range(test_data.index[0], test_data.index[0] + (data.index[-1] - data.index[0]), freq=freq))
    temp.loc[test_data.index] = test_data.values
    print("You have extended your test data using null values at a frequency of: {}".format(freq))
    return temp


# tensor reconstruction
def tensor_reconstruction(data: np.ndarray, folds: List[int], rank: int, decomposition_type: object = "parafac"):
    tensor = multifold(data, folds)
    if decomposition_type == "parafac":
        fac = parafac(tensor, rank=rank, n_iter_max=10000, tol=1.0e-15, linesearch=True)
        return tl.cp_to_tensor(fac).ravel(), np.sum([f.size for f in fac[1]][1:])
    elif decomposition_type == "tucker":
        if (type(rank) == int) or (type(rank) == float):
            rank = int(rank)
            rank = [rank for i in range(len(data.shape))]
        ranks = np.minimum(tensor.shape, rank)
        ranks[0] = 1
        core, factors = tucker(tensor, ranks=ranks, n_iter_max=10000, tol=1.0e-15)
        return tl.tucker_to_tensor((core, factors)).ravel(), np.sum(
            [ranks[i] * tensor.shape[i] for i in range(1, len(tensor.shape))]) + np.prod(ranks[1:])


def idct(w: np.ndarray, extr: int) -> np.ndarray:
    """
    Inverse DCT with extrapolation.

    :param w: series to apply IDCT (DCT-III)
    :param extr: number of time steps to extrapolate
    :return:
    """
    N = len(w)
    y = np.zeros(N + extr)

    for k in range(N):
        y[k] = w[0] + 2 * np.dot(
            w[1:], np.cos(np.pi * (2 * k + 1) * np.arange(1, N) / (2 * N))
        )

    y[-extr:] = y[:extr]

    return y / N / 2
