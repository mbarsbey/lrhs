import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, gammaln
from .utils import get_s_cp, get_s_cp_missing, get_s_cp_notmissing, lbeta_matrix, lbeta_scalar, rmse, mape

def vb(X: jnp.ndarray, 
           R: int,
           a: jnp.ndarray,
           b: jnp.ndarray = jnp.nan,
           num_epochs: int = 1,
           seed: int = 0,
           dir_smooth: jnp.ndarray = 0.1,
           print_freq=0
           ):
   if jnp.isnan(b):
      b = a / jnp.nansum(X)
   I = X.shape
   d = len(I)
   alpha_I = jnp.array([a/(R*I_n) for I_n in I])
   alpha_R = a / R

   S_sum = jnp.nansum(X)

   keys = jax.random.split(jax.random.PRNGKey(seed), d + 2)
   lkeys, key, rkey = keys[:-2], keys[-2], keys[-1]

   log_lam = digamma(S_sum + a) - jnp.log(b + 1)
   log_theta_R = jnp.log(jax.random.dirichlet(rkey, jnp.ones(R) * alpha_R + dir_smooth))
   log_theta_I = [jnp.log(jax.random.dirichlet(lkeys[n], jnp.ones(I_n) * alpha_I[n] + dir_smooth, [R])).T for n, I_n in enumerate(I)]
   log_C = a*jnp.log(b) - (a + S_sum)*jnp.log(b + 1) + gammaln(a + S_sum) - gammaln(a) - gammaln(X + 1).sum()                                       
   ELBO = (jnp.ones(num_epochs) * log_C).tolist()

   all_idx = jnp.array(jnp.where(X!=0)).T
   for i in range(num_epochs): 
      s_all, sp_all = get_s_cp(all_idx, log_theta_R, log_theta_I, X)
      S_I = [jax.ops.segment_sum(s_all, all_idx[:, n]).T for n in range(d)]
      S_R = s_all.sum(0)
      ELBO[i] -= sp_all.sum()
      
      for n, I_n in enumerate(I):
         log_theta_I[n] = digamma(S_I[n] + alpha_I[n]).T - digamma(S_R + alpha_R).T
         ELBO[i] += jnp.apply_along_axis(lbeta_matrix, -1, alpha_I[n] + S_I[n]).sum() - R * lbeta_scalar(alpha_I[n],I_n)
      log_theta_R = digamma(S_R + alpha_R) - digamma(S_sum + R * alpha_R)
      ELBO[i] += lbeta_matrix(alpha_R + S_R) - lbeta_scalar(alpha_R,R)
      if i and print_freq:
         if (i+1) % print_freq == 0:
            print(f"Epoch: {i+1}, ELBO: {ELBO[i]: .0f}")
   return jnp.array(ELBO), [log_lam, log_theta_R, *log_theta_I]


def vb_batch(X: jnp.ndarray, 
           R: int,
           a: jnp.ndarray,
           b: jnp.ndarray = jnp.nan,
           num_epochs: int = 1,
           seed: int = 0,
           dir_smooth: jnp.ndarray = 0.1,
           num_batches=0,
           print_freq=0):
   if jnp.isnan(b):
      b = a / jnp.nansum(X)
   I = X.shape
   d = len(I)
   alpha_I = jnp.array([a/(R*I_n) for I_n in I])
   alpha_R = a / R

   S_sum = jnp.nansum(X)

   keys = jax.random.split(jax.random.PRNGKey(seed), d + 2)
   lkeys, key, rkey = keys[:-2], keys[-2], keys[-1]

   log_lam = digamma(S_sum + a) - jnp.log(b + 1)
   log_theta_R = jnp.log(jax.random.dirichlet(rkey, jnp.ones(R) * alpha_R + dir_smooth))
   log_theta_I = [jnp.log(jax.random.dirichlet(lkeys[n], jnp.ones(I_n) * alpha_I[n] + dir_smooth, [R])).T for n, I_n in enumerate(I)]
   log_C = a*jnp.log(b) - (a + S_sum)*jnp.log(b + 1) + gammaln(a + S_sum) - gammaln(a) - gammaln(X + 1).sum()                                       
   ELBO = (jnp.ones(num_epochs) * log_C).tolist()

   all_idx = jnp.array(jnp.where(X!=0)).T
   for i in range(num_epochs): 
      if num_batches:
         S_R = jnp.zeros(R)
         S_I = [jnp.zeros_like(log_theta_I_n).T for log_theta_I_n in log_theta_I]
         sp_all = 0
         b = int((len(all_idx) / num_batches)+1)
         for j in range(num_batches):
            s_all, sp_all_  = get_s_cp(all_idx[j*b:(j+1)*b], log_theta_R, log_theta_I, X)
            S_R += s_all.sum(0)
            sp_all += sp_all_.sum()
            S_I = [S_I[n] + jax.ops.segment_sum(s_all, all_idx[j*b:(j+1)*b, n], num_segments=I[n]).T for n in range(d)]
      else:
         s_all, sp_all = get_s_cp(all_idx, log_theta_R, log_theta_I, X)
         S_I = [jax.ops.segment_sum(s_all, all_idx[:, n]).T for n in range(d)]
         S_R = s_all.sum(0)

      ELBO[i] -= sp_all.sum()
      
      for n, I_n in enumerate(I):
         log_theta_I[n] = digamma(S_I[n] + alpha_I[n]).T - digamma(S_R + alpha_R).T
         ELBO[i] += jnp.apply_along_axis(lbeta_matrix, -1, alpha_I[n] + S_I[n]).sum() - R * lbeta_scalar(alpha_I[n],I_n)
      log_theta_R = digamma(S_R + alpha_R) - digamma(S_sum + R * alpha_R)
      ELBO[i] += lbeta_matrix(alpha_R + S_R) - lbeta_scalar(alpha_R,R),
      if i and print_freq:
         if (i+1) % print_freq == 0:
            print(f"Epoch: {i+1}, ELBO: {ELBO[i]: .0f}")
   return jnp.array(ELBO), [log_lam, log_theta_R, *log_theta_I]


def vb_missing(X: jnp.ndarray, 
           R: int,
           a: jnp.ndarray,
           b: jnp.ndarray = jnp.nan,
           num_epochs: int = 1,
           seed: int = 0,
           print_freq=0,
           es_freq=100,
           es_epochs=500,
           X_gt=None,
           trend=None,
           es_idx=None,
           einsum=None,
           b_correction=False,
           dir_smooth: jnp.ndarray = 0.1,
           ):
   jax.config.update("jax_enable_x64", False)
   if jnp.isnan(b):
      b = a / (jnp.nansum(X)/(1-jnp.isnan(X).mean())) if b_correction else a / jnp.nansum(X)
   I = X.shape
   d = len(I)

   alpha_R = a / jnp.prod(R)
   alpha_I = jnp.array([(a/(R*I_n)) for n, I_n in enumerate(I)])
   Alpha_R = jnp.ones(R) * alpha_R
   Alpha_I = [jnp.ones((R, I[n])) * alpha_I_n for n, alpha_I_n in enumerate(alpha_I)]

   keys = jax.random.split(jax.random.PRNGKey(seed), d + 3)
   lkeys, key, rkey, rrkey = keys[:-3], keys[-3], keys[-2], keys[-1]
   S_sum = jnp.nansum(X)

   log_lam = jnp.log(jax.random.gamma(rrkey, a)/b) #digamma(S_sum + a) - jnp.log(b + 1)
   log_theta_R = jnp.log(jax.random.dirichlet(rkey, jnp.ones(R) * alpha_R + dir_smooth))
   log_theta_I = [jnp.log(jax.random.dirichlet(lkeys[n], jnp.ones(I_n) * alpha_I[n] + dir_smooth, [R])).T for n, I_n in enumerate(I)]

   ELBO = jnp.zeros(num_epochs).tolist()
   all_idx_missing = jnp.array(jnp.where(jnp.isnan(X))).T
   all_idx_nonmissing = jnp.array(jnp.where(~jnp.isnan(X))).T
   all_idx = jnp.vstack((all_idx_missing, all_idx_nonmissing))
   best_mape, best_rmse, best_epoch, best_params, es_evals = 1e6, 1e6, 0, [], []

   for i in range(num_epochs): 
      
      X_idx_m, s_all_m = get_s_cp_missing(all_idx_missing, log_lam, log_theta_R, log_theta_I, X)
      s_all, X_idx_elbo_nm = get_s_cp_notmissing(all_idx_nonmissing, log_lam, log_theta_R, log_theta_I, X)
      s_all = jnp.vstack((s_all_m, s_all))
      del s_all_m
      S_I = [jax.ops.segment_sum(s_all, all_idx[:, n]).T for n in range(d)]
      S_R = s_all.sum(0)
      S_sum = jnp.nansum(X) + X_idx_m.sum()
      ELBO[i] += X_idx_m.sum() + X_idx_elbo_nm.sum()
      ELBO[i] += a * jnp.log(b) - (S_sum + a) * jnp.log(b+1)
      for n in range(d):
         ELBO[i] +=  gammaln(Alpha_I[n] + S_I[n]).sum() - gammaln(Alpha_I[n]).sum()
      ELBO[i] += 2*gammaln(Alpha_R).sum() - 2*gammaln(Alpha_R + S_R).sum()
      ELBO[i] -= S_sum * log_lam + (S_R * log_theta_R).sum()
      for n in range(d):
         ELBO[i] -= (S_I[n] * log_theta_I[n].T).sum()
         
      log_lam = digamma(S_sum+a) - jnp.log(b+1.0)
      log_theta_R = digamma(S_R+Alpha_R) - digamma(S_sum+a)
      for n in range(d):
         log_theta_I[n] = digamma(S_I[n] + Alpha_I[n]).T - digamma(S_R + Alpha_R).T
      
      if i and es_freq and (i%es_freq==0):
         X_hat = np.einsum(einsum, *[jnp.exp(par) for par in [log_theta_R, *log_theta_I]])  * jnp.exp(log_lam)
         cur_rmse = rmse((X_hat + trend)[es_idx], X_gt[es_idx])
         cur_mape = mape((X_hat + trend)[es_idx], X_gt[es_idx])
         es_evals.append((i, cur_mape, cur_rmse, ELBO[i-1]))
         if i % print_freq == 0:
            print(f"Epoch: {i}, ELBO: {ELBO[i]: .0f}, Val MAPE: {es_evals[-1][1]: .6f}, Val RMSE: {es_evals[-1][2]: .6f}")
         if (cur_rmse < best_rmse) and (cur_mape < best_mape):
            best_rmse = cur_rmse
            best_mape = cur_mape
            best_epoch = i
            best_params = [arr.copy() for arr in [log_lam, log_theta_R, *log_theta_I]]
         else:
            if i - best_epoch >= es_epochs:
               print(f"Early stopping at epoch {i}, reverting to the best parameters obtained at epoch {best_epoch}.")
               log_lam, log_theta_R, *log_theta_I = best_params
               ELBO = ELBO[:best_epoch+1]
               break
   return jnp.array(ELBO), es_evals, [log_lam, log_theta_R, *log_theta_I]
