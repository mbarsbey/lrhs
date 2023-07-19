import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from typing import Union
from .utils import get_s_tucker

def vb_batch(X: jnp.ndarray, 
           R: Union[jnp.ndarray, int],
           a: jnp.ndarray,
           b: jnp.ndarray = jnp.nan,
           num_epochs: int = 1,
           seed: int = 0,
           dir_smooth: jnp.ndarray = 0.1,
           num_batches=0,
           print_freq=0,
           prev_thetas=None,
           prev_elbo=None):
   if jnp.isnan(b):
      b = a / jnp.nansum(X)
   I = X.shape
   d = len(I)
   R = jnp.array(R)

   alpha_R = a / jnp.prod(R)
   alpha_I = jnp.array([(a/(R[n]*I_n)) for n, I_n in enumerate(I)])
   Alpha_R = jnp.ones(R) * alpha_R
   Alpha_I = [jnp.ones((I[n], R[n])) * alpha_I_n for n, alpha_I_n in enumerate(alpha_I)]

   keys = jax.random.split(jax.random.PRNGKey(seed), d + 2)
   lkeys, key, rkey = keys[:-2], keys[-2], keys[-1]
   S_sum = jnp.nansum(X)
   log_C = a*jnp.log(b) - (a + S_sum)*jnp.log(b + 1) - gammaln(X + 1).sum()                                       
   if prev_thetas is not None:
      log_lam = prev_thetas[0]
      log_theta_R = prev_thetas[1]
      log_theta_I = prev_thetas[2:]
      start_epoch = len(prev_elbo)
      end_epoch = len(prev_elbo) + num_epochs
      ELBO = list(prev_elbo) + (jnp.ones(num_epochs) * log_C).tolist()
      print(f"Found previous params, starting from epoch {start_epoch}.")
   else:
      log_lam = digamma(S_sum + a) - jnp.log(b + 1)
      log_theta_R = jnp.log(jax.random.dirichlet(rkey, jnp.ones(R) * alpha_R + dir_smooth))
      log_theta_I = [jnp.log(jax.random.dirichlet(lkeys[n], jnp.ones(I_n) * alpha_I[n] + dir_smooth, [R[n]])).T for n, I_n in enumerate(I)]
      start_epoch = 0
      end_epoch = num_epochs
      ELBO = (jnp.ones(num_epochs) * log_C).tolist()
   all_idx = jnp.array(jnp.where(X!=0)).T
   for i in range(start_epoch, end_epoch):
      if num_batches:
        S_R = jnp.zeros(R)
        S_I = [jnp.zeros_like(log_theta_I_n) for log_theta_I_n in log_theta_I]
        sp_all = 0
        b = int((len(all_idx) / num_batches)+1)
        for j in range(num_batches):
          s_all, sp_all_  = get_s_tucker(all_idx[j*b:(j+1)*b], log_lam, log_theta_R, log_theta_I, X)
          S_R += s_all.sum(0)
          sp_all += sp_all_.sum()          
          del sp_all_
          for n in range(d):
            S_I[n] += jax.ops.segment_sum(s_all.sum(axis=1 + jnp.array([n_ for n_ in range(d) if n_ != n])), all_idx[j*b:(j+1)*b, n], num_segments=I[n])
          del s_all
      else:
        s_all, sp_all = get_s_tucker(all_idx, log_lam, log_theta_R, log_theta_I, X)
        S_I = [jax.ops.segment_sum(s_all.sum(axis=1 + jnp.array([n_ for n_ in range(d) if n_ != n])), all_idx[:, n], num_segments=I[n]) for n in range(d)]
        S_R = s_all.sum(0)
        del s_all
      ELBO[i] -= sp_all.sum()
      del sp_all
      S_R_ns = [S_R.sum(axis=[n_ for n_ in range(d) if n_ != n])  for n in range(d)]
      Alpha_R_ns = [Alpha_R.sum(axis=[n_ for n_ in range(d) if n_ != n])  for n in range(d)]
      ELBO[i] += gammaln(Alpha_R + S_R).sum() - gammaln(Alpha_R).sum()# parts that cancelled  + loggamma(a + S⁺) - loggamma(a)   + loggamma(a)  - loggamma(S⁺ + a)
      for n in range(d):
         ELBO[i] +=  gammaln(Alpha_I[n] + S_I[n]).sum() - gammaln(Alpha_R_ns[n] + S_R_ns[n]).sum() + gammaln(Alpha_R_ns[n]).sum() - gammaln(Alpha_I[n]).sum()
      log_theta_R = digamma(S_R+Alpha_R) - digamma(S_sum+a)
      for n in range(d):
         log_theta_I[n] = digamma(S_I[n] + Alpha_I[n]) - digamma(S_R_ns[n] + Alpha_R_ns[n]).T
      if i and print_freq:
         if (i+1) % print_freq == 0:
            print(f"Epoch: {i+1}, ELBO: {ELBO[i]: .0f}")
   return jnp.array(ELBO), [log_lam, log_theta_R, *log_theta_I]

def vb(X: jnp.ndarray, 
           R: Union[jnp.ndarray, int],
           a: jnp.ndarray,
           b: jnp.ndarray = jnp.nan,
           num_epochs: int = 1,
           seed: int = 0,
           dir_smooth: jnp.ndarray = 0.1,
           print_freq = 0,
           ):
   if jnp.isnan(b):
      b = a / jnp.nansum(X)
   I = X.shape
   d = len(I)
   R = jnp.array(R)
   
   alpha_R = a / jnp.prod(R)
   alpha_I = jnp.array([(a/(R[n]*I_n)) for n, I_n in enumerate(I)])
   Alpha_R = jnp.ones(R) * alpha_R
   Alpha_I = [jnp.ones((I[n], R[n])) * alpha_I_n for n, alpha_I_n in enumerate(alpha_I)]

   keys = jax.random.split(jax.random.PRNGKey(seed), d + 2)
   lkeys, key, rkey = keys[:-2], keys[-2], keys[-1]
   S_sum = jnp.nansum(X)

   log_lam = digamma(S_sum + a) - jnp.log(b + 1)
   log_theta_R = jnp.log(jax.random.dirichlet(rkey, jnp.ones(R) * alpha_R + dir_smooth))
   log_theta_I = [jnp.log(jax.random.dirichlet(lkeys[n], jnp.ones(I_n) * alpha_I[n] + dir_smooth, [R[n]])).T for n, I_n in enumerate(I)]

   log_C = a*jnp.log(b) - (a + S_sum)*jnp.log(b + 1) - gammaln(X + 1).sum()                                       
   ELBO = (jnp.ones(num_epochs) * log_C).tolist()

   all_idx = jnp.array(jnp.where(X!=0)).T
   for i in range(num_epochs):
      s_all, sp_all = get_s_tucker(all_idx, log_lam, log_theta_R, log_theta_I, X)
      
      S_I = [jax.ops.segment_sum(s_all.sum(axis=1 + jnp.array([n_ for n_ in range(d) if n_ != n])), all_idx[:, n]) for n in range(d)]
      S_R = s_all.sum(0)
      ELBO[i] -= sp_all.sum()

      S_R_ns = [S_R.sum(axis=[n_ for n_ in range(d) if n_ != n])  for n in range(d)]
      Alpha_R_ns = [Alpha_R.sum(axis=[n_ for n_ in range(d) if n_ != n])  for n in range(d)]
      
      ELBO[i] += gammaln(Alpha_R + S_R).sum() - gammaln(Alpha_R).sum()
      for n in range(d):
         ELBO[i] +=  gammaln(Alpha_I[n] + S_I[n]).sum() - gammaln(Alpha_R_ns[n] + S_R_ns[n]).sum() + gammaln(Alpha_R_ns[n]).sum() - gammaln(Alpha_I[n]).sum()
      log_theta_R = digamma(S_R+Alpha_R) - digamma(S_sum+a)
      for n in range(d):
         log_theta_I[n] = digamma(S_I[n] + Alpha_I[n]) - digamma(S_R_ns[n] + Alpha_R_ns[n]).T
      if i and print_freq:
         if (i+1) % print_freq == 0:
            print(f"Epoch: {i+1}, ELBO: {ELBO[i]: .0f}")
   return jnp.array(ELBO), [log_lam, log_theta_R, *log_theta_I]