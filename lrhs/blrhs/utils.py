import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.nn import logsumexp
from datetime import datetime
from collections.abc import Iterable
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json


lbeta_matrix = jax.jit(lambda X: gammaln(X).sum() - gammaln(X.sum()))
lbeta_scalar = jax.jit(lambda gamma, I: I*gammaln(gamma) - gammaln(I*gamma))

@jax.jit
def get_s_tucker(idx, log_lam, log_theta_R, log_theta_I, X):
   d, I = len(X.shape), X.shape
   log_p = log_lam + log_theta_R 
   for n in range(d):
      log_p += jnp.reshape(log_theta_I[n][idx[n], :], [1 if _n != n else log_theta_R.shape[_n] for _n in range(d)])
   log_p -= logsumexp(log_p)
   s = X[tuple(idx)] * jnp.exp(log_p)
   return s, s * log_p

get_s_tucker = jax.vmap(get_s_tucker, in_axes=[0, None, None, None, None])

@jax.jit
def get_s_cp(i, log_theta_R, log_theta_I, X):
    d = len(X.shape)
    log_p = log_theta_R + jnp.array([log_theta_I[n][i[n], :] for n in range(d)]).sum(0)
    log_p -= logsumexp(log_p)
    s = X[tuple(i)] * jnp.exp(log_p)
    return s, s * log_p

get_s_cp = jax.vmap(get_s_cp, in_axes=[0, None, None, None])

@jax.jit
def get_s_cp_missing(idx, log_lam, log_theta_R, log_theta_I, X):
    d, I = len(X.shape), X.shape
    log_p = log_lam + log_theta_R + jnp.array([log_theta_I[n][idx[n], :] for n in range(d)]).sum(0)
    log_p_idx = logsumexp(log_p)
    log_p -= log_p_idx
    return jnp.exp(log_p_idx), jnp.exp(log_p_idx) * jnp.exp(log_p)

get_s_cp_missing = jax.vmap(get_s_cp_missing, in_axes=[0, None, None, None, None])

@jax.jit
def get_s_cp_notmissing(idx, log_lam, log_theta_R, log_theta_I, X):
    d, I = len(X.shape), X.shape
    log_p = log_lam + log_theta_R + jnp.array([log_theta_I[n][idx[n], :] for n in range(d)]).sum(0)
    log_p_idx = logsumexp(log_p)
    log_p -= log_p_idx
    return X[tuple(idx)] * jnp.exp(log_p), X[tuple(idx)]*log_p_idx - gammaln(X[tuple(idx)] + 1.0)

get_s_cp_notmissing = jax.vmap(get_s_cp_notmissing, in_axes=[0, None, None, None, None])

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)    

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)   
    
def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

rmse = lambda y_hat, y: jnp.sqrt(((y_hat - y) ** 2).mean())

mape = lambda y_hat, y: (jnp.abs(y_hat - y)/y).mean()

def axis_except(ns, d):
  if not isinstance(ns, Iterable):
    ns = [ns]
  return tuple([i for i in range(d) if i not in ns])


def set_fonts(small, large):
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large)  
    plt.rc('axes', titlesize=large)    
    plt.rc('axes', labelsize=small) 
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)

argsort_ascending_2d = lambda X: (np.stack(np.unravel_index(np.flip(X.flatten().argsort()), X.shape)).T, jnp.flip(jnp.sort(X.flatten())))

def plot_weekly_patterns(P_GDH, idxs, values, figsize=(16,8), filepath=""):
  figsize=(3*len(idxs), 8)
  fig, axes = plt.subplots(1, len(idxs), figsize=figsize,sharey=True)
  for i, idx in enumerate(idxs):
    R_temp = P_GDH[tuple(idx)]
    R_temp /= R_temp.sum()
    axes[i].imshow(R_temp.T)
    if isinstance(idx, Iterable):
        axes[i].set_title("$\widehat{P}(r_{hour}=$" + f"${idx[0]+1}$" + "$,r_{day}=$" + f"${idx[1]+1})$" + f"\n$={values[i]: .2f}$")
    else:
        axes[i].set_title("$\widehat{P}(r=$" + f"${idx+1})$" + f"$={values[i]: .2f}$")
    axes[i].set_xticks([])
    axes[i].set_xlabel("Weekday")
    axes[i].set_xticks(range(7)); axes[i].set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    if not i:
      axes[i].set_ylabel("Hour")
      axes[i].set_yticks([-0.5, 5.5, 11.5, 17.5, 23.5]); axes[i].set_yticklabels(["00", "06", "12", "18", "24"])

  fig.tight_layout()

def plot_elbo(elbo):
   plt.figure(figsize=(10,2)); plt.plot(elbo); plt.xlabel("Epochs"); plt.ylabel("ELBO")

def _plot_bart_latents_through_seasons(total_commute, commute_morning, commute_evening, weekend_day, weekend_night, time, RW, other):
  fig, axes = plt.subplots(2, 1, figsize=(12,6), gridspec_kw={"height_ratios": [1, 3]}, sharex=True)
  ax = axes[0]
  ax.plot(total_commute)
  ax.set_ylabel("Total rides")
  ax.set_ylim(0, 3.5e7)
  ax.set_yticks([0,1.75e7, 3.5e7])
  ax.grid(which="major", alpha=0.5)
  ax = axes[1]
  ax.set_ylabel("Commute type ratios")
  ax.plot(commute_morning)
  ax.scatter(range(len(commute_morning)), commute_morning, label="Morning commute")
  ax.plot(commute_evening)
  ax.scatter(range(len(commute_evening)), commute_evening, label="Evening commute")
  ax.plot(weekend_day)
  ax.scatter(range(len(weekend_day)), weekend_day, label="Weekend daytime")
  ax.plot(weekend_night)
  ax.scatter(range(len(weekend_night)), weekend_night, label="Weekend night")
  ax.set_xticks(range(0, len(commute_morning)))
  ax.set_xticklabels([f"{time[i][0]}\nSpring" if i%4==0 else "" for i in range(len(commute_morning)) ])
  ax.grid(which="major", alpha=0.5)
  ax.legend(loc="upper left")
  fig.tight_layout()


def plot_bart_latents_through_seasons(X_temporal, P_RYWDH, start_year=2017, return_results=False):
  s_idx = {
      "winter": (list(range(9)), list(range(48,53))),
      "spring": list(range(10, 22)),
      "summer": list(range(22, 35)),
      "autumn": list(range(35, 48)),
  }
  total_commute = []
  commute_morning = []
  commute_evening = []
  weekend_day = []
  weekend_night = []
  time = []
  RW = []
  other = []
  start_year = start_year - 2011
  for i in range(start_year,12):
    for season in s_idx.keys():
      if (i == start_year) and (season == "winter"):
        continue
      if (i == 11) and (season not in ["winter", "spring"]):
        continue
      if season == "winter":
        P_RDH_YW = P_RYWDH[:, :, :, :, slice(i-1,i), s_idx[season][0], :, :].sum(axis=(1, 4, 5,6,7)) + P_RYWDH[:, :, :, :, slice(i,i+1), s_idx[season][1], :, :].sum(axis=(1, 4, 5,6,7))
        total_commute.append(X_temporal[slice(i-1,i), s_idx[season][0], :, :].sum() + X_temporal[slice(i,i+1), s_idx[season][1], :, :].sum())
      else:
        P_RDH_YW = P_RYWDH[:, :, :, :, slice(i,i+1), s_idx[season], :, :].sum(axis=(1, 4, 5,6,7))
        total_commute.append(X_temporal[slice(i,i+1), s_idx[season], :, :].sum())
      RW.append(P_RDH_YW.sum(axis=(1,2))/P_RDH_YW.sum(axis=(1,2)).sum())
      P_RDH_YW = P_RDH_YW.sum(0)
      P_RDH_YW /= P_RDH_YW.sum()
      commute_evening.append(P_RDH_YW[1, 3])
      commute_morning.append(P_RDH_YW[1, 1])
      weekend_day.append(P_RDH_YW[0, 2])
      weekend_night.append(P_RDH_YW[0, 0])
      other.append(P_RDH_YW[0, 1] + P_RDH_YW[0, 3] + P_RDH_YW[1, 0] + P_RDH_YW[1, 2])
      time.append((2011+i, season))
  _plot_bart_latents_through_seasons(total_commute, commute_morning, commute_evening, weekend_day, weekend_night, time, RW, other)
  if return_results:
    return total_commute, commute_morning, commute_evening, weekend_day, weekend_night, time, RW, other
  

weekly_reshape = lambda x: np.concatenate((x.flatten(), np.array([np.nan] * 288))).reshape(9, 7, 144)
def gz_fold_day(X):
    X_day = np.zeros((214, 9, 7, 144))
    X_day[:,:,:,:] = np.nan
    for i in range(214):
        X_day[i, :, :, :] = weekly_reshape(X[i])
    return X_day, "r, ir, jr, kr, lr -> ijkl"

weekly_hourly_reshape = lambda x: np.concatenate((x.flatten(), np.array([np.nan] * 288))).reshape(9, 7, 24, 6)
def gz_fold_day_hour(X):
    X_day = np.zeros((214, 9, 7, 24, 6))
    X_day[:,:,:,:] = np.nan
    for i in range(214):
        X_day[i, :, :, :] = weekly_hourly_reshape(X[i])
    return X_day, "r, ir, jr, kr, lr, mr -> ijklm"

def get_X_imputation_experiment(data_folder="data/guangzhou/", missing_scenario="random", missing_ratio=0.5, num_total_dimensions=3, es_ratio=0.01):
    X = jnp.array(loadmat(data_folder + 'tensor.mat')["tensor"])
    if missing_scenario == "fiber":
        random_matrix = jnp.array(loadmat(data_folder + 'random_matrix.mat')["random_matrix"])
        binary_tensor = np.zeros_like(X)
        test_binary_tensor = np.zeros_like(X)
        for i in range(binary_tensor.shape[0]):
            for j in range(binary_tensor.shape[1]):
                binary_tensor[i,j,:] = jnp.round(random_matrix[i, j]+0.5-missing_ratio-es_ratio);
                test_binary_tensor[i,j,:] = jnp.round(random_matrix[i, j]+0.5-missing_ratio);
    if missing_scenario == "random":
        random_tensor = jnp.array(loadmat(data_folder + 'random_tensor.mat')["random_tensor"])
        binary_tensor = jnp.round(random_tensor+0.5-missing_ratio-es_ratio);
        test_binary_tensor = jnp.round(random_tensor+0.5-missing_ratio);
    X_censored = X * binary_tensor
    X_censored = X_censored.at[X_censored==0].set(jnp.nan)
    X = X.at[X==0].set(jnp.nan)
    einsum = "r, ir, jr, kr -> ijk"
    assert num_total_dimensions in [3, 4, 5]
    if num_total_dimensions > 3:
      fold = gz_fold_day if num_total_dimensions == 4 else gz_fold_day_hour
      X, einsum = fold(X)
      X_censored, einsum = fold(X_censored)
      test_binary_tensor, einsum = fold(test_binary_tensor)
      binary_tensor, einsum = fold(binary_tensor)
    es_idx = ~jnp.isnan(X) & (test_binary_tensor != 0) & (binary_tensor == 0)
    test_idx = ~jnp.isnan(X) & (test_binary_tensor == 0)
    return X_censored, X, es_idx, test_idx, einsum

def get_trend(X, detrending_type):
    if detrending_type == "min":
        return np.nanmin(np.array(X), axis=tuple([i for i in range(1, len(X.shape))]), keepdims=True)
    elif detrending_type == "min":
        return 0
    else:
        raise NotImplementedError("The detrending method you selected is not implemented.")
