
from abc import ABC, abstractproperty, abstractmethod
import copy
import collections
import itertools
import gym
from gym import Wrapper
import inspect
import logging
import numpy as np
import os
from omegaconf import OmegaConf
import omegaconf
import pdb
import scipy
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as F
import torch_optimizer as optim
from torch.distributions.kl import kl_divergence as t_kl_divergence
from torch.distributions import Normal, Independent, OneHotCategorical, Categorical, MixtureSameFamily, RelaxedOneHotCategorical
import time
import tqdm
import warnings

# In-package imports.
import util.class_util as classu
import util.ml_util as mlu
import util.ptu as ptu

np.set_printoptions(precision=1, suppress=True)
log = logging.getLogger(os.path.basename(__file__))

ALL_KLDS_ANALYTIC = {1: True}
MC_N = 100
CATEGORICAL_PARAM_MODE = 'probs'

class DetachableDict(dict):
    def detach_(self):
        for v in self.values():
            if v is not None: v.detach_()

class Timer(object):
    def __init__(self, description, mode='info', enabled=False):
        self.description = description
        assert(hasattr(log, mode))
        self.print_fn = getattr(log, mode)
        self.enabled = enabled

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        if self.enabled:
            self.print_fn(f"{self.description:<30}: {self.end-self.start:.2f}")

class MemberCachingContext:
    def __init__(self, inst, member_name):
        self.inst = inst
        self.member_name = member_name

    def __enter__(self):
        # Grab a reference to the member
        self.member_cached = getattr(self.inst, self.member_name)

    def __exit__(self, type, value, traceback):
        # Set the instance's member reference to be the original reference. Allows for restoring the member after the context exits.
        setattr(self.inst, self.member_name, self.member_cached)

class PreserveRolloutState(MemberCachingContext):
    def __init__(self, inst):
        """This context will preserve the rollout_state member of the models, to enable computation of forking futures (e.g. one-step and multistep priors)"""
        super().__init__(inst, member_name='rollout_state')

def kl_divergence(a, b):
    try:
        return t_kl_divergence(a, b)
    except NotImplementedError:
        warnings.warn(f"KLD between {type(a)} and {type(b)} not implemented! Will estimate it with samples instead")
        ALL_KLDS_ANALYTIC[1] = False
        return MC_kld(a, b, N=MC_N)

def MC_kld(gmm_f, gmm_g, N=50):
    f_samples = gmm_f.sample((N,))
    try:
        negentropy = -1 * gmm_f.entropy()
    except NotImplementedError:
        negentropy = gmm_f.log_prob(f_samples)
    if N < 10:
        log.warning(f"GMM N is small={N}!")
        
    # Make sure the log-prob gmm is actually a mixture distribution (otherwise we should just use normal KLD)
    # assert(isinstance(gmm_g, MixtureSameFamily))
    cross_entropy = - gmm_g.log_prob(f_samples)
    MC_kld = torch.mean(negentropy + cross_entropy, 0)
    return MC_kld

def create_k_sized_slices_of_dim(arr, k, dim):
    """Create all k-sized slices along dim and put them in a new axis"""
    tuples = []
    d = arr.shape[dim]
    for i in range(k):
        end = d - k + i + 1
        # To enable arbitrary dim slicing, use the JIT version.
        # https://discuss.pytorch.org/t/use-python-like-slice-indexing-across-a-given-dimension/89606/8
        tuples.append(torch.ops.aten.slice(arr, dim, i, end, 1))
    return torch.stack(tuples, dim+1)

def xe(a, b):
    # KL = XE - H -> XE = KL + H
    return kl_divergence(a, b) + entropy(a)

def kl_divergence_biased_gradient(a, b, a_frozen, b_frozen, sampling_dist_bias=0.5):
    assert(0. <= sampling_dist_bias <= 1.)
    if sampling_dist_bias == 0.5:
        return kl_divergence(a, b)
    else:
        return sampling_dist_bias * kl_divergence(a, b_frozen) + (1 - sampling_dist_bias) * kl_divergence(a_frozen, b)

def kl_divergence_entropy_bias(a, b, entropy_bias=0.8):
    # KL(a,b) = H(a, b) - H(a)
    # KL_bias(a, b; bias) = 2 * (1 - entropy_bias) * H(a,b) - 2 * entropy_bias * H(a)
    # KL_bias(a, b; 0.5) = H(a,b) - H(a) = KL(a,b)
    # KL_bias(a, b; 0.8) = 0.4 * H(a,b) - 1.6 * H(a)
    # KL_bias(a, b; eps) ~= 2 * H(a,b)
    # KL_bias(a, b; 1-eps) ~= -2 * H(a)
    assert(0. < entropy_bias < 1.)
           
    if np.allclose(ptu.to_numpy(entropy_bias), 0.5):
        # KL_bias(a, b; 0.5) = H(a,b) - H(b)
        return kl_divergence(a, b)
    else:
        # KL_bias(a, b; bias) = 2 * (1 - entropy_bias) * H(a,b) - 2 * entropy_bias * H(a)
        cross_entropy_mx = 2 * (1 - entropy_bias)
        entropy_mx = 2 * entropy_bias
        a_samples = a.rsample((MC_N,))
        # Estimate cross entropy by sampling.
        cross_entropy = -1 * b.log_prob(a_samples).mean(0)
        entropy = a.entropy()
        return cross_entropy_mx * cross_entropy - entropy_mx * entropy

def entropy(dist):
    try:
        return dist.entropy()
    except NotImplementedError:
        return MC_entropy(dist)

def MC_entropy(dist, N=5):
    return -dist.log_prob(dist.sample((N,))).mean()

class LatentStateDistribution(ABC):
    def __init__(self, do_build_rsample=False, is_belief=False):
        self._weights = None
        self.is_belief = is_belief
        self.detached = False
        if do_build_rsample: self.build_rsample()        

    def detach_(self):
        self.detached = True
        
    @abstractproperty
    def _distribution(self):
        pass

    @abstractproperty
    def _distribution_detached(self):
        pass    

    @property
    def distribution(self):
        # If it's detached, redirect references to this property
        if self.detached: return self.distribution_detached
        res = getattr(self, '_distribution_', None)
        if res is None: setattr(self, '_distribution_', self._distribution)
        return getattr(self, '_distribution_')

    @property
    def distribution_detached(self):
        res = getattr(self, '_distribution_detached_', None)
        if res is None: setattr(self, '_distribution_detached_', self._distribution_detached)
        return getattr(self, '_distribution_detached_')
    
    @abstractproperty
    def B(self): pass    

    def rsample(self):
        return dict(latent_dist_sample=self.distribution.rsample())

    def flatten_sample(self, sample):
        return sample.view(self.distribution.batch_shape + (np.prod(self.distribution.event_shape),))

    def build_rsample(self):
        self.state_sample = self.distribution.rsample()

    def set_previous_particle_weights(self, weights):
        assert(self._weights is None)
        if weights is None:
            weights = 1/self.K * torch.ones((self.B, self.K), device=ptu.device)
        self._weights = weights
        assert(self._weights.ndim == 2)
        assert(self._weights.shape == (self.B, self.K))

    @property
    def belief_distribution(self):
        if self.is_belief:
            assert(self.K == 1)
            return self.distribution_detached
        else:
            assert(self.ndim in (3, 4))
            assert(self._weights is not None)

            # The weights correspond to the beliefs of the previous particles. We'll build an estimate of the belief
            #  distribution by taking an expectation of the latent state distribution under the previous beliefs
            categorical_weights = self._weights        
            categorical = Categorical(categorical_weights, validate_args=True)
            # b(z_t) \approx sum_k weight_k p(z_t)
            # E.g. for the posterior:
            # b(z_t) \approx sum_k b(z_{t-1}^k)/K q(z_t | z_{t-1}^k, a_{t-1}, o_t), by using the previous particle weights.
            with torch.no_grad():
                dist = MixtureSameFamily(mixture_distribution=categorical, component_distribution=self.distribution_detached, validate_args=True)
            return dist

    @abstractproperty
    def belief_distribution_parameters(self):
        pass
    
class NormalLatentStateDistribution(LatentStateDistribution):
    @classu.member_initialize
    def __init__(self, state_mean, state_stddev, mixture=False, do_build_rsample=False, is_belief=False):
        super().__init__(do_build_rsample=do_build_rsample, is_belief=is_belief)
        self._T = None
        if state_mean.ndim == state_stddev.ndim == 1:
            self._D = state_mean.shape[0]
            self._B = None            
            self._K = None
            self.ndim = 1
        elif state_mean.ndim == state_stddev.ndim == 2:
            self._B, self._D = state_mean.shape
            self._K = None
            self.ndim = 2
        elif state_mean.ndim == state_stddev.ndim == 3:
            self._B, self._K, self._D = state_mean.shape
            self.ndim = 3
        elif state_mean.ndim == state_stddev.ndim == 4:
            self._T, self._B, self._K, self._D = state_mean.shape
        else:
            raise ValueError("ndim")

    def __repr__(self):
        return f"{self.__class__.__name__}(T={self._T}, B={self._B}, K={self._K}, D={self._D}, is_belief={self.is_belief})"

    @property
    def shape(self):
        return (self.T, self.B, self.K, self.D)

    @property
    def T(self): return self._T
    
    @property
    def K(self): return self._K
        
    @property
    def B(self): return self._B

    @property
    def D(self): return self._D
    
    @property
    def _distribution(self):
        # 2D diagonal gaussian
        return Independent(Normal(loc=self.state_mean, scale=self.state_stddev, validate_args=True), reinterpreted_batch_ndims=1, validate_args=True)

    @property
    def _distribution_detached(self):
        # 2D diagonal gaussian        
        return Independent(Normal(loc=self.state_mean.detach(), scale=self.state_stddev.detach(), validate_args=True), reinterpreted_batch_ndims=1, validate_args=True)    

    @property
    def distribution_parameters(self):
        with torch.no_grad():
            return torch.cat((self.state_mean, self.state_stddev), axis=-1)

    @property
    def belief_distribution_parameters(self):
        assert(self.B == 1)
        if self.is_belief:            
            return self.distribution_parameters
        else:
            mean = self.distribution.mean.reshape((self.B, -1))
            stddev = self.distribution.stddev.reshape((self.B, -1))
            weights = self._weights 
            return torch.cat((mean, stddev, weights), axis=-1)

class StraightThroughOneHotCategorical(OneHotCategorical):
    def rsample(self, *args, **kwargs):
        return self.sample(*args, **kwargs) + self.probs - self.probs.detach()

class IndependentCategoricalLatentStateDistribution(LatentStateDistribution):
    @classu.member_initialize
    def __init__(self, state_logits, mode='st', do_build_rsample=False, is_belief=False):
        super().__init__(do_build_rsample=do_build_rsample)
        # (B, K, E, D)
        assert(state_logits.ndim == 4)
        self._B = state_logits.shape[0]
        self._K = state_logits.shape[1]
        self._E = state_logits.shape[2]
        self._D = state_logits.shape[3]
        self.ndim = state_logits.ndim
        # NB if this is too small, things can go nan.
        self.temperature = 1. * torch.ones((self._K,), device=ptu.device)
        self.is_belief = is_belief
        with torch.no_grad():
            if CATEGORICAL_PARAM_MODE == 'logits':
                self._dist_params = self.state_logits.view((self.B, self._K, self._E * self._D))
            elif CATEGORICAL_PARAM_MODE == 'probs':
                # Maybe it's easier to learn from the dist params if they're normalized first?
                self._dist_params = torch.softmax(self.state_logits, -1).view((self.B, self._K, self._E * self._D))
            else:
                raise ValueError(f"Unknown categorical param mode {CATEGORICAL_PARAM_MODE}")
        assert(mode in ('relaxed', 'st'))
        self.mode = mode

    @property
    def K(self): return self._K

    @property
    def B(self): return self._B

    @property
    def D(self): return self._D * self._E

    @property
    def _distribution(self):
        if self.mode == 'relaxed':
            return Independent(RelaxedOneHotCategorical(temperature=self.temperature, logits=self.state_logits, validate_args=True), 1)
        else:
            return Independent(StraightThroughOneHotCategorical(logits=self.state_logits, validate_args=False), reinterpreted_batch_ndims=1, validate_args=False)
    @property
    def _distribution_detached(self):
        if self.mode == 'relaxed':        
            return Independent(RelaxedOneHotCategorical(temperature=self.temperature, logits=self.state_logits.detach(), validate_args=False), 1)
        else:
            return Independent(StraightThroughOneHotCategorical(logits=self.state_logits.detach(), validate_args=False), reinterpreted_batch_ndims=1, validate_args=False)

    @property
    def distribution_parameters(self):
        return self._dist_params

    @property
    def belief_distribution_parameters(self):
        assert(self.B == 1)
        if self.is_belief:
            return self.distribution_parameters
        else:
            params = self.distribution_parameters.reshape((self.B, -1))
            weights = self._weights 
            return torch.cat((params, weights), axis=-1)

class BeliefFilteringSequence:
    @classu.member_initialize
    def __init__(self,
                 observation_model,
                 prior_latent_dists,
                 posterior_latent_dists,
                 prior_model,
                 posterior_model,
                 particles,
                 importance_components,
                 observation_images_TBO=None,
                 prior_latent_dists_multistep=None,
                 prior_latent_dists_multistep_backward=None,
                 multistep_distribution_forecast=False,
                 multistep_observation_forecast=False,
                 observation_beforecast=False,
                 image_loss_scale=1.,
                 multistep_freeze_posteriors=True,
                 **kwargs):

        if self.observation_images_TBO is None:
            self.observation_images_TBO = []
        if self.prior_latent_dists_multistep is None:
            self.prior_latent_dists_multistep = []
        if self.prior_latent_dists_multistep_backward is None:
            self.prior_latent_dists_multistep_backward = []            
        if self.observation_beforecast:
            raise NotImplementedError

        assert(len(self.prior_latent_dists) == len(self.posterior_latent_dists))

        self.actions = [None]
        self.agent_poses = []
        self._finalized = False

    @property
    def rollout_state(self):
        """Return all state components that the -----> belief filtering step <-------- depends upon.

        So it's not currently a complete state representation of this object, just sufficient for belief_filter_single_step!
        """

        # Store rollout state in a detachable dict so we can use it to stop-grad and initialize a new sequence.
        rstate = DetachableDict(particles=self.particles[-1], weights_t_normalized=self.importance_components[-1]['weights_t_normalized'])
        rstate['current_prior_dist'] = self.prior_latent_dists[-1]
        rstate['current_posterior_dist'] = self.posterior_latent_dists[-1]
        rstate['prior'] = self.prior_model.rollout_state
        rstate['posterior'] = self.posterior_model.rollout_state
        return rstate

    @rollout_state.setter
    def rollout_state(self, rstate):
        # Import the rollout state.
        self.particles[-1] = rstate['particles']
        self.importance_components[-1]['weights_t_normalized'] = rstate['weights_t_normalized']
        self.prior_model.rollout_state = rstate['prior']
        self.posterior_model.rollout_state = rstate['posterior']
        self.prior_latent_dists[-1] = rstate['current_prior_dist']
        self.posterior_latent_dists[-1] = rstate['current_posterior_dist']
    
    def compute_belief_filter_losses_t(self, beta, alpha, include_diagnostics=False):
        assert(beta is not None)
        assert(alpha is not None)
        ic_t = self.importance_components[-1]
        po_t = self.posterior_latent_dists[-1]
        pr_t = self.prior_latent_dists[-1]
        log_obs_t = ic_t['log_obs']

        # If alpha is 0.5, use normal KL div.
        use_biased_kl = not torch.isclose(alpha, 0.5 * torch.ones_like(alpha)).item()
        if use_biased_kl:
            kl = kl_divergence_biased_gradient(po_t.distribution, pr_t.distribution, po_t.distribution_detached, pr_t.distribution_detached, sampling_dist_bias=alpha)
        else:
            kl = kl_divergence(po_t.distribution, pr_t.distribution)

        log_obs_mean = log_obs_t.mean()
        kl_mean = kl.mean()
        weighted_kl_mean = beta * kl_mean

        # ELBO(q,p) = E_{q(z|x)} log p(x|z) - KL(q(z|x) || p(z))
        ELBO_t = self.image_loss_scale * log_obs_mean - weighted_kl_mean

        losses = dict(total_loss_t=-1 * ELBO_t,
                      kl_t=kl_mean.detach(),
                      log_obs_t=log_obs_mean.detach(),
                      weighted_kl_t=weighted_kl_mean.detach())

        stats = dict()
        if include_diagnostics:
            with torch.no_grad():
                pdb.set_trace()
                prior_obs_dist = self.observation_model.forward(pr_t.detached.view((-1, self.D)), batch_shape=(self.B, self.K))
                prior_obs_mse = F.mse_loss(prior_obs_dist.mean, self.observation_images_TBO[:, :, None])
                posterior_obs_mean = torch.stack([_['obs_likelihood_t'].mean for _ in self.importance_components]).view((self.T, self.B, self.K) + self.obs_shape)
                posterior_obs_mse = F.mse_loss(posterior_obs_mean, self.observation_images_TBO[:, :, None])
                stats.update(dict(prior_obs_mse=prior_obs_mse, posterior_obs_mse=posterior_obs_mse))
        return {**losses, **stats}
    
    def __len__(self):
        return len(self.posterior_latent_dists)

    def finalize(self, with_posterior_observation_dists, with_prior_observation_dists):
        assert(not self._finalized)

        self.B = self.posterior_latent_dists[0].B
        self.K = self.posterior_latent_dists[0].K
        self.D = self.posterior_latent_dists[0].D
        self.T = len(self.posterior_latent_dists)
        
        self.log_K_pt = torch.log(torch.tensor(self.K, device=ptu.device, dtype=torch.float32))
        
        if isinstance(self.observation_images_TBO, list):
            self.observation_images_TBO = torch.stack(self.observation_images_TBO, axis=0)

        self.obs_shape = self.observation_images_TBO.shape[2:]            
        
        if len(self.prior_latent_dists_multistep) == 0:
            self.prior_latent_dists_multistep = [[]] * len(self.prior_latent_dists)
        if len(self.prior_latent_dists_multistep_backward) == 0:
            self.prior_latent_dists_multistep_backward = [[]] * len(self.prior_latent_dists)            

        assert(_.B == self.B for _ in self.posterior_latent_dists)
        assert(_.B == self.B for _ in self.prior_latent_dists)
        assert(self.observation_images_TBO.shape[:2] == (self.T, self.B))
        assert(len(self.prior_latent_dists) == self.T)
        assert(len(self.posterior_latent_dists) == self.T)
        assert(len(self.prior_latent_dists_multistep) == self.T)
        assert(len(self.prior_latent_dists_multistep_backward) == self.T)
        assert(len(self.agent_poses) == self.T)
        self._finalized = True

    def sample_posterior_beliefs(self):
        return [_.belief_distribution.sample() for _ in self.posterior_latent_dists]

    def sample_prior_beliefs(self):
        return [_.belief_distribution.sample() for _ in self.prior_latent_dists]
            
    @property
    def posteriors(self):
        assert(self._finalized)
        return [_.distribution for _ in self.posterior_latent_dists]
    
    @property
    def priors(self):
        assert(self._finalized)
        return [_.distribution for _ in self.prior_latent_dists]

    @property
    def posteriors_detached(self):
        assert(self._finalized)
        return [_.distribution_detached for _ in self.posterior_latent_dists]
    
    @property
    def priors_detached(self):
        assert(self._finalized)
        return [_.distribution_detached for _ in self.prior_latent_dists]

    def compute_belief_filter_losses(self, beta, alpha, loss_mode, normalize=False, include_diagnostics=True):
        assert(loss_mode in ('smc', 'elbo', 'regular', 'batch'))
        
        with Timer('losses asserts'):
            assert(self._finalized or loss_mode == 'batch')
            assert(isinstance(loss_mode, str))
            assert(isinstance(beta, torch.Tensor))
            assert(isinstance(alpha, torch.Tensor))

        z_TfBK = torch.stack([_.state_sample for _ in self.posterior_latent_dists], 0)
        if self.observation_model.k_steps > 1:
            # Slice off the z that we don't have the labels for.
            z_TfBK = z_TfBK[:-self.observation_model.k_steps+1]

        TfBK_shape = z_TfBK.shape[:3]
        z_batch = z_TfBK.reshape((-1, self.posterior_model.latent_state_size))
        if None not in self.agent_poses:
            agent_poses_TBP = torch.stack(self.agent_poses, 0)
            agent_poses_batch = agent_poses_TBP.view((-1, self.observation_model.agent_pose_size))
        else:
            agent_poses_batch = None

        # B, T, O_shape
        oBTO = self.observation_images_TBO.transpose(0, 1)

        # not implementend > 1
        assert(self.K == 1)

        future_action_batch = None
        # No need to create larger image labels.
        obs_target = oTfBKE = oBTO.transpose(0,1).unsqueeze(2)

        # -- Run the obs model forward --
        obs_likelihood = self.observation_model.forward(z_batch, future_actions=future_action_batch, agent_pose=agent_poses_batch, batch_shape=TfBK_shape)

        # Ensure label dimensionality is correct.                
        assert(oTfBKE.shape[3:] == obs_likelihood.event_shape)

        # Ensure batch dimensionality is correct.                            
        assert(oTfBKE.shape[:3] == obs_likelihood.batch_shape == TfBK_shape)
        log_obs = obs_likelihood.log_prob(obs_target)

        # If alpha is 0.5, use normal KL div.
        use_biased_kl = not torch.isclose(alpha, 0.5 * torch.ones_like(alpha)).item()
        if use_biased_kl:
            kl = torch.stack([kl_divergence_biased_gradient(po, pr, po_fr, pr_fr, sampling_dist_bias=alpha) for (po, pr, po_fr, pr_fr) in
                              zip(self.posteriors, self.priors, self.posteriors_detached, self.priors_detached)])                
        else:
            kl = torch.stack([kl_divergence(po, pr) for (po, pr) in zip(self.posteriors, self.priors)])

        log_obs_mean = log_obs.mean()
        kl_mean = kl.mean()
        weighted_kl_mean = beta * kl_mean

        # ELBO(q,p) = E_{q(z|x)} log p(x|z) - KL(q(z|x) || p(z))
        ELBO = log_obs_mean - weighted_kl_mean

        multistep_kls = []
        multistep_log_obs = []
        multistep_kls_backward = []

        for target_obs, posterior, posterior_frozen, multistep_priors in zip(self.observation_images_TBO, self.posteriors, self.posteriors_detached, self.prior_latent_dists_multistep):
            if len(multistep_priors):
                # Removed unused code for clarity.
                raise NotImplementedError
            multistep_kls.append(0.0)
            multistep_log_obs.append(0.)

        for target_posterior, multistep_priors_backward in zip(self.posteriors, self.prior_latent_dists_multistep_backward):
            if len(multistep_priors_backward):
                # Removed unused code for clarity.
                raise NotImplementedError
            else:
                multistep_kls_backward.append(0.0)

        # Average over T.
        multistep_kl_backward_mean = sum(multistep_kls_backward) / len(multistep_kls_backward)
        multistep_kl_mean = sum(multistep_kls) / len(multistep_kls)
        multistep_log_obs_mean = self.image_loss_scale * sum(multistep_log_obs) / len(multistep_log_obs)

        weighted_multistep_kl_mean = beta * multistep_kl_mean
        weighted_multistep_kl_backward_mean = beta * multistep_kl_backward_mean
        total_loss = -1 * ELBO - multistep_log_obs_mean + weighted_multistep_kl_mean + weighted_multistep_kl_backward_mean
        log.debug(f"ELBO: {ELBO:.2g}, log_obs: {log_obs_mean:.2g}, weighted_kl: {weighted_kl_mean:.2g}, " +
                  f"multistep_log_obs: {multistep_log_obs_mean:.2g}, weighted_multistep_kl: {weighted_multistep_kl_mean:.2g}")
        losses = dict(total_loss=total_loss,
                      kl=kl_mean,
                      log_obs=log_obs_mean,
                      weighted_kl=weighted_kl_mean,
                      multistep_log_obs=multistep_log_obs_mean,
                      multistep_kl=multistep_kl_mean,
                      weighted_multistep_kl=weighted_multistep_kl_mean,
                      weighted_multistep_kl_backward=weighted_multistep_kl_backward_mean)

        stats = dict(posterior_entropy=(sum([_.entropy() for _ in self.posteriors])/self.T).mean(),
                     prior_entropy=(sum([_.entropy() for _ in self.priors])/self.T).mean())

        if include_diagnostics:
            with torch.no_grad():
                prior_latent_means = torch.stack([_.mean for _ in self.priors_detached]).view((-1, self.D))
                prior_obs_dist = self.observation_model.forward(prior_latent_means, agent_pose=agent_poses_batch, batch_shape=(self.T, self.B, self.K))
                prior_obs_mse = F.mse_loss(prior_obs_dist.mean[:obs_target.shape[0]], obs_target)
                posterior_obs_mse = F.mse_loss(obs_likelihood.mean, obs_target)
                stats.update(dict(prior_obs_mse=prior_obs_mse, posterior_obs_mse=posterior_obs_mse))
        return {**losses, **stats}

    def visualize(self):
        assert(self.B == 1)
        with torch.no_grad():
            vis = [self.observation_images_TBO,]

            if None not in self.agent_poses:
                agent_poses_TBP = torch.stack(self.agent_poses, 0)
                agent_poses_batch = agent_poses_TBP.view((-1, self.observation_model.agent_pose_size))
            else:
                agent_poses_batch = None
            vis.append(self.observation_model.forward(torch.stack([_.flatten() for _ in self.sample_posterior_beliefs()]), agent_pose=agent_poses_batch).sample((1,)).transpose(1,0))
            vis.append(self.observation_model.forward(torch.stack([_.flatten() for _ in self.sample_prior_beliefs()]), agent_pose=agent_poses_batch).sample((1,)).transpose(1,0))

            if self.observation_model.k_steps > 1:
                # Clip out extra image predictions.
                vis[-1] = vis[-1][:, :, 0]
                vis[-2] = vis[-2][:, :, 0]

            assert(all([_.ndim == 5 for _ in vis]))
            # (T, B, C, H*3, W)
            viscat = torch.cat(vis, axis=-2)
            # (B, C, H*3, W, T)
            viscat = viscat.permute((1, 2, 3, 0, 4))
            # (B, C, H*3, W*T)
            viscat = viscat.reshape(viscat.shape[:-2]+(-1,))
            # (B, ..., C)
            return torch.clamp(viscat.permute((0, 2, 3, 1)), 0., 1.)[..., :3]

    def __eq__(self, other):
        debug = True
        
        with torch.no_grad():
            prior_lens_eq = len(self.priors) == len(other.priors)
            posterior_lens_eq = len(self.posteriors) == len(other.priors)

            # noise_eq = all([torch.isclose(a,b).all().item() for (a,b) in zip(self.prior_noise_samples, other.prior_noise_samples)])
            prior_klds = [kl_divergence(mp, op).sum() for (mp, op) in zip(self.priors, other.priors)]
            posterior_klds = [kl_divergence(mp, op).sum() for (mp, op) in zip(self.posteriors, other.posteriors)]

            # Increased the atol because LowRankMVN kl_div to itself can exhibit numerical issues
            zero = torch.tensor(0.)
            priors_eq = torch.allclose(sum(prior_klds), zero, atol=1e-1)
            posteriors_eq = torch.allclose(sum(posterior_klds), zero, atol=1e-1)

            if debug:
                assert(prior_lens_eq)
                assert(posterior_lens_eq)
                assert(priors_eq)
                assert(posteriors_eq)
            return prior_lens_eq and posterior_lens_eq and priors_eq and posteriors_eq
        
class VBFModel(torch.nn.Module):
    """Combines a prior, posterior, observation featurizer, and observation model: 
        
        posterior=q(z_t | o_{<=t}, a_{<t})
        prior=p(z_t | o_{<t}, a_{<t})
        observation_model=p(o_t|z_t)
        observation_featurizer= o_t = f(Image_t)

        Use it to produce BeliefFilteringSequences: lists of posteriors, priors, and observation models (distributions) given data
    """

    @property
    def posterior_model(self):
        return self._posterior_model

    @property
    def prior_model(self):
        return self._prior_model

    @property
    def observation_model(self):
        return self._observation_model

    @property
    def observation_featurizer(self):
        return self._observation_featurizer

    @classmethod
    def from_env_and_kwargs(cls, env, model_kwargs):
        observation_space = env.observation_space[model_kwargs.observation_key]
        observation_shape = observation_space.low.shape
        obs_dim = int(np.prod(observation_shape))

        action_space_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # For now, we'll make action repr env-dependent.
        model_kwargs['one_hotify_actions'] = action_space_discrete
        
        if action_space_discrete:
            assert(model_kwargs['one_hotify_actions'])
            action_dim = env.action_space.n
        else:
            assert(not model_kwargs['one_hotify_actions'])
            action_dim = int(env.action_space.shape[0])
            
        return cls(model_kwargs=model_kwargs, action_dim=action_dim, obs_shape=observation_shape, obs_dim=obs_dim) 
        
    @classu.member_initialize
    def __init__(self, model_kwargs, action_dim, obs_shape, obs_dim, models={}):
        """Instantiate model.

        :param model_kwargs: 
        :param action_dim: 
        :param obs_shape: 
        :param obs_dim: 
        """
        super().__init__()
        self._init_kwargs = dict(model_kwargs=model_kwargs, action_dim=action_dim, obs_shape=obs_shape, obs_dim=obs_dim)


        # Don't modify the incoming dict.
        if isinstance(model_kwargs, omegaconf.dictconfig.DictConfig):
            model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
        assert(isinstance(model_kwargs, dict))
        
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.observation_key = self.model_kwargs['observation_key']

        # Store some kwargs
        self.one_hotify_actions = model_kwargs['one_hotify_actions']
        
        self.model_kwargs['prior_model']['action_size'] = action_dim
        self.model_kwargs['posterior_model']['action_size'] = action_dim

        if self.model_kwargs['loss']['compute_online']:
            assert(not self.model_kwargs['loss']['multistep_distribution_forecast'])
            assert(not self.model_kwargs['loss']['multistep_observation_forecast'])
            assert(not self.model_kwargs['loss']['multistep_observation_forecast'])
            # Sanity check. Prevent silent non-truncation if we're computing online (otherwise, we should compute losses offline).
            assert(self.model_kwargs['loss']['truncation_horizon'] < self.model_kwargs['horizon'])

        self._require_importance_components = self.model_kwargs['training_particle_resampling'] or self.model_kwargs['testing_particle_resampling']
            
        self.t = 0

        if 'observation_model' in models:
            self._observation_model = models.pop('observation_model')
        else:
            self._observation_model = VisualObservationModel(**self.model_kwargs['observation_model']).to(device=ptu.device)
        if 'observation_featurizer' in models:
            self._observation_featurizer = models.pop('observation_featurizer')
        else:
            self._observation_featurizer = VisualObservationFeaturizer(**self.model_kwargs['observation_featurizer']).to(device=ptu.device)

        if 'sequential_observation_featurizer' in models:
            self._sequential_observation_featurizer = models.pop('sequential_observation_featurizer')
        else:
            sof = self.model_kwargs['sequential_observation_featurizer'].lower()
            if sof == 'independent':
                self._sequential_observation_featurizer = IndependentSequentialObservationFeaturizer(self._observation_featurizer)
            else:
                raise ValueError(sof)

        # Store the classes and init kwargs to enable SWAGification later.
        self._prior_model_class, self._prior_model_init_kwargs = get_class_and_kwargs_of_prior(self.model_kwargs['prior_model'])
        
        if 'prior_model' in models:
            self._prior_model = models.pop('prior_model')
        else:
            self._prior_model = self._prior_model_class(**self._prior_model_init_kwargs).to(device=ptu.device)

        self._posterior_model_class, self._posterior_model_init_kwargs = get_class_and_kwargs_of_posterior(self.model_kwargs['posterior_model'])

        if 'posterior_model' in models:
            self._posterior_model = models.pop('posterior_model')
        else:
            self._posterior_model = self._posterior_model_class(**self._posterior_model_init_kwargs).to(device=ptu.device)

        # The featurization of the belief distribution needs to be consistent.
        if self.posterior_model.is_belief:
            assert(self.model_kwargs['augmentation_mode'] == 'mean_stddev')
        else:
            assert(self.model_kwargs['augmentation_mode'] == 'mixture_mean_stddev')

        # Ensure all provided models have been consumed.
        assert(len(models) == 0)
        
        self._backward_priors = []
        log.info(f"Created {self}")

    def get_models(self):
        # Return a dictionary of components, useful for instantiating other VBFModels e.g. in an ensemble.
        return {'observation_model': self._observation_model,
                'observation_featurizer': self._observation_featurizer,
                'sequential_observation_featurizer': self._sequential_observation_featurizer,
                'prior_model': self._prior_model,
                'posterior_model': self._posterior_model}
    
    @property
    def mixture_belief_distribution_size(self):
        K = self.model_kwargs['K_particles_infer']
        return K * self.latent_distribution_size + K

    @property
    def gaussian_belief_distribution_size(self):
        return self.latent_distribution_size

    @property
    def belief_distribution_size(self):
        if self.model_kwargs['augmentation_mode'] == 'mixture_mean_stddev':
            return self.mixture_belief_distribution_size
        elif self.model_kwargs['augmentation_mode'] == 'mean_stddev':
            return self.gaussian_belief_distribution_size
        elif self.model_kwargs['augmentation_mode'] == 'ekf_mean_stddev':
            return self.gaussian_belief_distribution_size
        else:
            raise ValueError(self.model_kwargs['augmentation_mode'])

    @property
    def latent_distribution_size(self):
        return self.posterior_model.latent_distribution_size

    @property
    def latent_state_size(self):
        return self.posterior_model.latent_state_size    

    def __repr__(self):
        return f"{self.__class__.__name__}({self.prior_model}, {self.posterior_model}, {self.observation_model}, {self._observation_featurizer})"
        
    def belief_filter_reset(self, B, K, obs_0, obs_0_enc=None, track_internally=True, agent_pose_0=None):
        # Initialize prior and posterior.
        prior_0 = self.prior_model.get_initial_latent_dists(B=B, K=K)
        posterior_0 = self.posterior_model.get_initial_latent_dists(B=B, K=K)

        if self.model_kwargs['observe_agent_pose']:
            assert(agent_pose_0 is not None)
        
        for bp in self._backward_priors: bp.get_initial_latent_dists(B=B, K=K)
            
        self._sequential_observation_featurizer.reset()
        assert(0 <= obs_0.max().item() <= 1.)

        if obs_0_enc is None:
            if obs_0.shape == self.obs_shape:
                obs_0_enc = self._sequential_observation_featurizer.online(obs_0)
            elif obs_0.shape[1:] == self.obs_shape:
                obs_0_enc = self._sequential_observation_featurizer.online(obs_0)
            else:
                raise ValueError("obs shape")
            
        if obs_0.ndim == 4:
            assert(obs_0.shape[0] == B)
        else:
            raise ValueError("obs0 dim wrong")

        importance_components_0 = self._compute_importance_components(prior_0, posterior_0, posterior_0.state_sample, obs_0)
        bfs = BeliefFilteringSequence(observation_model=self._observation_model,
                                      particles=[posterior_0.state_sample],
                                      importance_components=[importance_components_0],
                                      prior_model=self.prior_model,
                                      posterior_model=self.posterior_model,
                                      prior_latent_dists=[prior_0],
                                      posterior_latent_dists=[posterior_0],
                                      observation_images_TBO=[obs_0],
                                      **self.model_kwargs['loss'])
        
        bfs.prior_latent_dists[0].set_previous_particle_weights(None)
        bfs.posterior_latent_dists[0].set_previous_particle_weights(None)
        bfs.agent_poses.append(agent_pose_0)

        B, K = bfs.particles[0].shape[:2]

        # (B, K) of batch inds. [[0_0, 0_1, ... 0_K], [1_0, 1_1, ... 1_K],  ... [(B-1)_1, ... (B-1)_K]]
        bfs.particle_batch_inds = torch.arange(B, device=ptu.device)[:, None].repeat((1, K))
        
        # Store the current encoded observation.
        bfs.o_tm1_enc = obs_0_enc
        if track_internally: self.current_bfs = bfs            
        return bfs

    def _get_action_repr(self, a):
        if isinstance(a, int):
            a = torch.tensor([[a]], device=ptu.device)
            
        if self.one_hotify_actions:
            if a.shape[-1] == 1:
                a = torch.nn.functional.one_hot(a[..., 0].to(torch.int64), num_classes=self.action_dim)
            elif a.shape[-1] == self.action_dim:
                pass
            else:
                raise ValueError
        else:
            pass

        return a.to(torch.float32)

    def _prepare_single_observation_and_action(self, o_t, a_tm1, o_t_enc=None):
        # Batch of vectors
        assert(a_tm1.ndim == 2)
        assert(o_t is not None)
        assert(0. <= o_t.max().item() <= 1.)
        
        if o_t_enc is None:
            assert(o_t.shape == self.obs_shape)
            o_t = o_t[None]
            o_t_enc = self._sequential_observation_featurizer.online(o_t)

        assert(o_t_enc.ndim >= 2)

        a_tm1 = self._get_action_repr(a_tm1)
        return (a_tm1, o_t_enc, o_t)
    
    def belief_filter_single_step(self, o_t, a_tm1, o_t_enc=None, bfs_t=None, bfs_rollout_state=None, a_t_to_k=[], t=None, update_bfs=True, agent_pose_t=None):
        """Step posterior, prior, and other metadata of the current belief filtering sequence.

        :param o_t: (...) + obs_shape
        :param a_tm1: (..., D)
        :param o_t_enc: (..., E)
        :returns: 
        :rtype: 

        """
        if len(a_t_to_k): assert(t is not None)

        # If the bfs_t is not provided, use one stored in the state of the model.
        if bfs_t is None: bfs_t = self.current_bfs
        if bfs_rollout_state is None: bfs_rollout_state = bfs_t.rollout_state

        if t is not None:
            # Ensure that we have all of the previous priors and posteriors in the bfs for t in [1, T]
            assert(t > 0)
            assert(t == len(bfs_t.prior_latent_dists))
            assert(t == len(bfs_t.posterior_latent_dists))

        # Create action reprs, create encoding if its missing, check shapes, etc.
        a_tm1, o_t_enc, o_t = self._prepare_single_observation_and_action(o_t=o_t, a_tm1=a_tm1, o_t_enc=o_t_enc)

        # Step the particles forward through the prior and posterior.
        belief_filtering_step = self._step_particles(
            particles_tm1=bfs_rollout_state['particles'], particle_weights_tm1=bfs_rollout_state['weights_t_normalized'],
            o_t=o_t, o_t_enc=o_t_enc,
            a_tm1=a_tm1,
            particle_batch_inds=bfs_t.particle_batch_inds,
            agent_pose_t=agent_pose_t)

        if len(a_t_to_k):
            # Use the updated particles to step the latent dynamics model extra steps
            latent_dists_forecast = self.forecast_latent_dist_with_action_plan(belief_filtering_step['particles'], a_t_to_k)
            for tau, ldf in enumerate(latent_dists_forecast): bfs_t.prior_latent_dists_multistep[t + tau].append(ldf)

        if update_bfs:
            bfs_t.importance_components.append(belief_filtering_step['importance_components'])
            bfs_t.particles.append(belief_filtering_step['particles'])
            # Record the new prior and posterior.
            bfs_t.prior_latent_dists.append(belief_filtering_step['prior'])
            bfs_t.posterior_latent_dists.append(belief_filtering_step['posterior'])
            bfs_t.observation_images_TBO.append(o_t)
            bfs_t.actions.append(a_tm1)
            bfs_t.agent_poses.append(agent_pose_t)

            # Store the current encoded observation.
            bfs_t.o_tm1_enc = o_t_enc
        return belief_filtering_step

    def _compute_importance_components(self, prior_latent_dist_t, posterior_latent_dist_t, latent_state_samples_t, o_t):
        if not self._require_importance_components:
            return dict(weights_t_normalized=None)
        else:
            # For SMC things.
            B, K = latent_state_samples_t.shape[:2]
            D = self.latent_state_size

            # Batch them because the observation model demands 2D inputs.
            latent_state_samples_t_batched = posterior_latent_dist_t.state_sample.view((B*K, D))
            # Compute the observation likelihood distribution, reshape the dimensionality to allow for correct batch log-prob calculation.
            obs_likelihood_t = self.observation_model.forward(latent_state_samples_t_batched, batch_shape=(B,K))
            # (B, K). Compute observation log-likelihoods of current observation.
            obs_log_prob_t = obs_likelihood_t.log_prob(o_t[:, None])
            # (B, K). Computer prior log probability of posterior's samples
            prior_log_prob_t = prior_latent_dist_t.distribution.log_prob(latent_state_samples_t)
            # (B, K). Compute posterior log probability of posterior's samples
            posterior_log_prob_t = posterior_latent_dist_t.distribution.log_prob(latent_state_samples_t)
            # (B, K). Compute the log importance weights.
            log_weight_t = prior_log_prob_t + obs_log_prob_t - posterior_log_prob_t

            # The unnormalized log weights can be extremely largely negative and cause torch.exp to evaluate to 0. So let's not compute them this way.
            # weights_t = torch.exp(log_weight_t)

            # Perform the weight normalization in log-space:
            #   Compute log w - log sum (exp log (w)) = log (w / sum(exp log(w))) = log(w / sum(w)).
            log_of_normalized_weights = log_weight_t - torch.logsumexp(log_weight_t, -1)[..., None]

            # Compute the normalized weights.
            weights_t_normalized = torch.exp(log_of_normalized_weights)

            # Fix numerical issues, they may not precisely sum to 1 yet.
            weights_t_normalized /= weights_t_normalized.sum(-1)[..., None]

            return dict(log_prior=prior_log_prob_t,
                        log_obs=obs_log_prob_t,
                        log_posterior=posterior_log_prob_t,
                        log_weight=log_weight_t,
                        log_of_normalized_weights_t=log_of_normalized_weights,
                        weights_t_normalized=weights_t_normalized,
                        obs_likelihood_t=obs_likelihood_t)

    def _resample(self, weights_t_normalized, latent_state_samples_t, particle_batch_inds):
        B, K, *event_shape = latent_state_samples_t.shape
        B_subset = weights_t_normalized.shape[0]
        
        # These two lines should perform equivalent sampling operations.
        # resampled_particle_inds = Categorical(probs=weights_t).sample((K,)).permute(1,0).flatten()
        # (B, K) -> (BK,)
        resampled_particle_inds = torch.multinomial(weights_t_normalized.detach(), K, replacement=True).flatten()
        
        # (B, K) + Eshape. Extract K particles for each of the entries in the batch.
        resampled_particles = latent_state_samples_t[particle_batch_inds, resampled_particle_inds].view((B_subset, K) + tuple(event_shape))
        return resampled_particles

    def _step_particles(self, particles_tm1, particle_weights_tm1, o_t, o_t_enc, a_tm1, particle_batch_inds, agent_pose_t):
        """

        :param particles_tm1: (B, K, D)
        :param o_t_enc: 
        :param a_tm1: 
        :returns: 
        :rtype: 

        """
        assert(particles_tm1.ndim >= 3)
        assert(o_t.ndim == 4)
        assert(o_t_enc.ndim >= 2)
        assert(a_tm1.ndim == 2)

        B, K = particles_tm1.shape[:2]
        observation_t_tile = K_tile(K, o_t_enc)

        # The outgoing particle weights from this function are normalized; treat incoming weights as normalized too.
        particle_weights_tm1_normalized = particle_weights_tm1

        particle_resampling = self.model_kwargs['training_particle_resampling'] if self.training else self.model_kwargs['testing_particle_resampling']

        if not particle_resampling:
            particles_tm1_post_resampled = particles_tm1
        elif self.model_kwargs['conditionally_resample_particles']:
            raise NotImplementedError("Account for unresampled particle weights")
        else:
            # Resample all particles from ancestors. L7 of Alg 1 in http://proceedings.mlr.press/v84/naesseth18a/naesseth18a.pdf
            particles_tm1_post_resampled = self._resample(weights_t_normalized=particle_weights_tm1_normalized, latent_state_samples_t=particles_tm1, particle_batch_inds=particle_batch_inds.flatten())

        # Clean up pre-resampled particles so we don't accidentally use them.
        del particles_tm1

        # Package the resampled particles and the previous actions.
        za_tm1_and_agent_pose_t = dict(latent_dist_sample_tm1=particles_tm1_post_resampled, action_tm1=a_tm1, agent_pose_t=agent_pose_t)

        # Compute f(. | ancestor, ...)
        prior_latent_dist_t = self.prior_model.forward(**za_tm1_and_agent_pose_t)

        posterior_kwargs = {**za_tm1_and_agent_pose_t, **dict(prior_latent_dist_t=prior_latent_dist_t, prior_rollout_state_t=self.prior_model.rollout_state)}

        # Compute g(. | ancestor, ...)
        posterior_latent_dist_t = self.posterior_model.forward(observation_t=observation_t_tile, **posterior_kwargs)

        # Extract particles now. (x_t of Alg 1 http://proceedings.mlr.press/v84/naesseth18a/naesseth18a.pdf)
        particles_t = posterior_latent_dist_t.state_sample
        
        # Set the previous particle weights of the new posterior and prior.
        posterior_latent_dist_t.set_previous_particle_weights(particle_weights_tm1_normalized)
        prior_latent_dist_t.set_previous_particle_weights(particle_weights_tm1_normalized)

        # (B, K) Compute the importance weights of particles with resampled ancestors. We'll use these directly in the ELBO SMC lower-bound.
        importance_components_t = self._compute_importance_components(prior_latent_dist_t=prior_latent_dist_t,
                                                                      posterior_latent_dist_t=posterior_latent_dist_t,
                                                                      latent_state_samples_t=particles_t,
                                                                      o_t=o_t)
        return dict(prior=prior_latent_dist_t,
                    posterior=posterior_latent_dist_t,
                    particles=particles_t,
                    importance_components=importance_components_t)

    def forecast_latent_dist_with_action_plan(self, particles_t, a_t_to_k=[]):
        """
        Compute the multistep distributions.

        :param particles_t: z_t, e.g. from q(z_t | o<=t, a<t)
        :param a_t_to_k: a_{t:t+K}
        :returns: 
        :rtype: 

        """

        particles_t_plus_tau = particles_t
        latent_dist_forecast = []
        # Make sure not to mess up the rollout state of the prior when forecasting.
        with PreserveRolloutState(self.prior_model):
            for tau in range(0, len(a_t_to_k)):
                action_t_plus_tau = self._get_action_repr(a_t_to_k[tau])
                # Package the resampled particles and the previous actions.
                z_and_a_t_plus_tau = dict(latent_dist_sample_tm1=particles_t_plus_tau, action_tm1=action_t_plus_tau)
                next_prior = self.prior_model.forward(**z_and_a_t_plus_tau)
                next_prior.build_rsample()
                particles_t_plus_tau = next_prior.state_sample
                latent_dist_forecast.append(next_prior)
        return latent_dist_forecast
        
    def belief_filter(self,
                      K=None,
                      batch=None,
                      observations=None,
                      actions=None,
                      actions_are_previous=False,
                      with_prior_observation_dists=False,
                      with_posterior_observation_dists=False,
                      compute_losses_online_and_truncate_BPTT=False,
                      optimizer_step_and_reset_func=None,
                      run_multistep=True,
                      include_diagnostics=False,
                      beta=None,
                      alpha=None,
                      memory={}):
        """Infer with a batch containing observations and actions, or pass them separately.

        Compute all b(s_t) for t \in [1, T]

        :param batch: must contain observations and actions
        :param observations: (B, T, D)
        :param actions: (B, T, A)
        :returns: 
        :rtype: 

        """
        log.debug(f"In {inspect.currentframe().f_code.co_name} of {self.__class__.__name__} instance.")
        if K is None:
            K = self.model_kwargs['K_particles_train']
        
        if observations is None or actions is None:
            assert(batch is not None)
            observations = batch['observations']
            actions = batch['actions']

        B, T = observations.shape[:2]
        if self.model_kwargs['observe_agent_pose']:
            # T, B, P
            agent_poses = batch['agent_pose'].transpose(0, 1)
            assert(agent_poses is not None)
        else:
            agent_poses = None

        # obs_TBO, obs_enc_TBO, actions, nonterminals_0toTm1 = self.prepare_observations_and_actions(observations, actions)
        actions = self.prepare_actions(actions)
        obs_TBO = observations.transpose(0, 1)

        is_start = batch.get('is_start', None)

        if is_start is not None and not is_start:
            # If we're partway through belief-filtering a trajectory, pull in the old rollout state
            rollout_state = memory['rollout_state']
            # Detach the rollout state, truncating the gradients.
            rollout_state.detach_()
            # Reset the filtering sequence
            bfs_t = self.belief_filter_reset(B, K=K, obs_0=obs_TBO[0], obs_0_enc=None, track_internally=False, agent_pose_0=agent_poses[0] if agent_poses is not None else None)
            # Update the state of the filtering sequence
            bfs_t.rollout_state = rollout_state
        else:
            bfs_t = self.belief_filter_reset(B, K=K, obs_0=obs_TBO[0], obs_0_enc=None, track_internally=False, agent_pose_0=agent_poses[0] if agent_poses is not None else None)

        # Allocate lists of multistep dists.
        bfs_t.prior_latent_dists_multistep = [[] for _ in range(T)]
        bfs_t.prior_latent_dists_multistep_backward = [[] for _ in range(T)]

        # Step through.        
        for t in range(1, T):
            # When d==1, there's no multistep.
            d = self.model_kwargs['multistep_distance'] if run_multistep else 1
            # Don't use actions beyond the penultimate (because we have no observation supervision beyond the ultimate)
            multistep_actions = actions[t:min(t+d-1, T-1)]
            o_t_enc = self._sequential_observation_featurizer.online(obs_TBO[t])
            agent_pose_t = None if agent_poses is None else agent_poses[t]
            step_input = dict(o_t=obs_TBO[t], o_t_enc=o_t_enc, a_tm1=actions[t-1], bfs_t=bfs_t, t=t, a_t_to_k=multistep_actions, agent_pose_t=agent_pose_t)
            self.belief_filter_single_step(**step_input)

            if run_multistep:
                for (k, bp) in enumerate(self._backward_priors):
                    if compute_losses_online_and_truncate_BPTT: raise NotImplementedError
                    start_idx = t
                    target_idx = t - 1 - k
                    if target_idx < 0: break
                    z_start = bfs_t.particles[start_idx]
                    inp = dict(latent_dist_sample_tm1=z_start, action_tm1=actions[start_idx - 1])
                    bfs_t.prior_latent_dists_multistep_backward[target_idx].append(bp.forward(**inp))
        bfs_t.finalize(with_posterior_observation_dists=with_posterior_observation_dists, with_prior_observation_dists=with_prior_observation_dists)
        memory['rollout_state'] = bfs_t.rollout_state
        return bfs_t

    def prepare_observations_and_actions(self, observations, actions):
        batch_size, T, *single_obs_shape = observations.shape

        assert(tuple(single_obs_shape) == self.obs_shape)
        obs_BTO = observations
        
        nonterminals_0toTm1 = None
        
        # (T, B, ...)
        obs_TBO = obs_BTO.transpose(0, 1)
        # (T, B, A)
        actions = actions.transpose(0, 1)

        # potentially one hotify etc.
        actions = self._get_action_repr(actions)

        self._sequential_observation_featurizer.reset()
        obs_enc_TBO = self._sequential_observation_featurizer.batch(obs_TBO)
        
        assert(obs_enc_TBO.shape[0] == obs_TBO.shape[0])
        return obs_TBO, obs_enc_TBO, actions, nonterminals_0toTm1

    def prepare_actions(self, actions):
        # (T, B, A)
        actions = actions.transpose(0, 1)

        # potentially one hotify etc.
        actions = self._get_action_repr(actions)
        return actions

    def belief_filter_and_compute_losses(self, batch=None, observations=None, actions=None, actions_are_previous=False, with_prior_observation_dists=False,
                                         compute_losses_online_and_truncate_BPTT=False,
                                         optimizer_step_and_reset_func=None,
                                         t=None, loss_mode=None, beta=None, alpha=None, include_diagnostics=True):
        log.debug(f"In {inspect.currentframe().f_code.co_name} of {self.__class__.__name__} instance.")

        beta = torch.tensor(beta, device=ptu.device)
        alpha = torch.tensor(alpha, device=ptu.device)

        with Timer('forward'):
            belief_filtering_sequence = self.belief_filter(
                batch=batch, observations=observations, actions=actions, actions_are_previous=actions_are_previous,
                with_posterior_observation_dists=True,
                compute_losses_online_and_truncate_BPTT=compute_losses_online_and_truncate_BPTT,
                with_prior_observation_dists=with_prior_observation_dists,
                include_diagnostics=include_diagnostics,
                optimizer_step_and_reset_func=optimizer_step_and_reset_func,
                beta=beta,
                alpha=alpha)

        if t is None:
            t = self.t
        else:
            self.t = t

        if not compute_losses_online_and_truncate_BPTT:
            with Timer('compute belief filter losses'):
                bfl = belief_filtering_sequence.compute_belief_filter_losses(beta=beta, alpha=alpha, loss_mode=loss_mode, include_diagnostics=include_diagnostics)
        else:
            bfl = belief_filtering_sequence.online_losses
        return bfl

    def randomized_evaluation(self, replay_buffer, N_minibatches, B, H, beta=1.0, alpha=0.5, loss_mode='regular'):
        all_losses = []
        with torch.no_grad():        
            for n in range(N_minibatches):
                data = replay_buffer.random_batch_window(batch_size=B, window_len=H, obs_key=self.observation_key)
                data = ptu.np_to_pytorch_batch(data)
                all_losses.append(self.belief_filter_and_compute_losses(batch=data, with_prior_observation_dists=True, beta=beta, alpha=alpha, loss_mode=loss_mode))
        extract_values_for_key = lambda k: [ptu.to_numpy(_[k]) for _ in all_losses]
        compute_stats = lambda arr: (np.mean(arr), scipy.stats.sem(arr, axis=None))

        keys = list(all_losses[0].keys())
        stats_dict = {k: compute_stats(extract_values_for_key(k)) for k in keys}
        return stats_dict

    replay_buffer_eval = randomized_evaluation

    def test_batch_vs_sequence(self, batch):
        observations = batch['observations']
        actions = batch['actions']
        K = self.model_kwargs['K_particles_train']

        if self.model_kwargs['observe_agent_pose']:
            # T, B, P
            agent_poses = batch['agent_pose'].transpose(0, 1)
            assert(agent_poses is not None)
        else:
            agent_poses = None
        
        B, T = observations.shape[:2]
        obs_TBO, obs_enc_TBO, actions, nonterminals_0toTm1 = self.prepare_observations_and_actions(observations, actions)

        obs_1toT = obs_TBO[1:]
        obs_enc_1toT = obs_enc_TBO[1:]
        actions_0toTm1 = actions[:-1]
        _1toT = list(range(1, T+1))

        if not ALL_KLDS_ANALYTIC[1]:
            log.warning("Cannot compare sequences due to imprecision in KLD computation")
            return

        with torch.no_grad():        
            with torch.random.fork_rng(devices=[ptu.device]):
                # Multistep requires random sampling -- since the single-step currently doesn't run_multistep, don't run_multistep the batch either.
                bfs_batch = self.belief_filter(batch=batch, with_prior_observation_dists=True, with_posterior_observation_dists=True, run_multistep=False)

            with torch.random.fork_rng(devices=[ptu.device]):
                # Multistep requires random sampling -- since the single-step currently doesn't run_multistep, don't run_multistep the batch either.
                bfs_batch2 = self.belief_filter(batch=batch, with_prior_observation_dists=True, with_posterior_observation_dists=True, run_multistep=False)

            with torch.random.fork_rng(devices=[ptu.device]):
                agent_pose_0 = None if agent_poses is None else agent_poses[0]
                bfs_t = self.belief_filter_reset(B, K=K, obs_0=obs_TBO[0], track_internally=False, agent_pose_0=agent_pose_0)
                for t, o_t, o_t_enc, a_tm1 in zip(_1toT, obs_1toT, obs_enc_1toT, actions_0toTm1):
                    agent_pose_t = None if agent_poses is None else agent_poses[t]
                    self.belief_filter_single_step(o_t=o_t, a_tm1=a_tm1, o_t_enc=o_t_enc, t=t, bfs_t=bfs_t, agent_pose_t=agent_pose_t)
                bfs_t.finalize(with_posterior_observation_dists=True, with_prior_observation_dists=True)

        assert(bfs_batch == bfs_batch2)
        assert(bfs_batch == bfs_t)

    def replay_buffer_test_batch_vs_sequence(self, replay_buffer, N, B, H):
        data = replay_buffer.random_batch_window(batch_size=B, window_len=H, obs_key=self.observation_key)
        data = ptu.np_to_pytorch_batch(data)
        return self.test_batch_vs_sequence(data)        

    def replay_buffer_vis(self, replay_buffer, H):
        with torch.no_grad():
            self.eval()
            data = replay_buffer.random_batch_window(batch_size=1, window_len=H, obs_key=self.observation_key)
            # Convert numpy arrays to torch.Tensors on the default device.
            data = ptu.np_to_pytorch_batch(data)
            bfs = self.belief_filter(batch=data, with_prior_observation_dists=True, with_posterior_observation_dists=True)
            if type(bfs) == list:
                model_vis = torch.clamp(bfs[0].visualize(), 0., 1.)
            else:
                model_vis = torch.clamp(bfs.visualize(), 0., 1.)
        return ptu.to_numpy(model_vis[0])

class PosteriorModel(ABC):
    @abstractproperty
    def latent_state_size(self): pass

    @abstractproperty
    def latent_distribution_size(self): pass

    @abstractproperty
    def action_size(self): pass    

    @abstractproperty
    def observation_size(self): pass

    @abstractmethod
    def forward(self, latent_dist_tm1, observation_t, action_tm1, prior_latent_dist_t, prior_rollout_state_t, agent_pose_t):
        pass

    @abstractmethod
    def get_initial_latent_dists(self, B, K):
        pass

    @abstractproperty
    def rollout_state(self):
        """This is used to hold temporal information over time, in order to be able to reset to a specific rollout state properly"""                                                 

class PriorModel(ABC):
    @abstractmethod
    def get_initial_latent_dists(self, B, K):
        pass
    
    @abstractproperty
    def latent_state_size(self): pass

    @abstractproperty
    def latent_distribution_size(self): pass

    @abstractproperty
    def action_size(self): pass
    
    @abstractproperty
    def forward(self, latent_dist_sample_tm1, action_tm1, agent_pose_t):
        pass

    @abstractproperty
    def rollout_state(self):
        """This is used to hold temporal information over time, in order to be able to reset to a specific rollout state properly"""

def build_mlp(input_size, hidden_sizes, output_size, activation_layer_class=nn.ELU, output_activation=None):
    sizes = list(zip([input_size] + hidden_sizes, hidden_sizes + [output_size]))
    layers = []
    for (s0, s1) in sizes[:-1]:
        layers.append(nn.Linear(s0, s1))
        layers.append(activation_layer_class())
    layers.append(nn.Linear(*sizes[-1]))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def get_act(act, max_=None):
    act = act.lower()
    if act == 'softplus':
        return torch.nn.Softplus()
    elif act == 'exp':
        if max_ is None:
            # Clamp so that exp output isn't larger than 1 billion.
            # max_ = 20.7233
            # Clamp so that exp output isn't larger than 1 thousand. Assuming you're just going to use this for
            #  computing stddevs, in which case you probably don't want them to be larger than 1 thousand.
            max_ = 7.
        return lambda x: torch.exp(torch.clamp(x, max=max_))
    else:
        raise ValueError(act)

#------------------ PRIORS -------------------------    
class MLPNormalPriorModel(PriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, min_std_dev=1e-2, hidden_sizes=[200], std_act='softplus', embed_action=False, abs_mean_range=1e3, **kwargs):
        super().__init__()
        self._abs_mean_range = abs_mean_range
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = 2 * self.latent_state_size
        self._action_size = action_size
        self.embed_action = embed_action
        if self.embed_action:
            self.fc_embed_action = nn.Linear(self.action_size, self.latent_state_size)
            self._input_size = self.latent_state_size + self.fc_embed_action.out_features
        else:
            self.fc_embed_action = lambda x: x
            self._input_size = self.latent_state_size + self.action_size
        self.fc_latent_dist_params = build_mlp(self._input_size, hidden_sizes, self.latent_distribution_size)
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.std_act = get_act(std_act)

    # Stateless rollout w.r.t. inputs to forward.
    @property
    def rollout_state(self): return DetachableDict()
    
    @rollout_state.setter
    def rollout_state(self, x): pass
        
    @property
    def K(self): return self._K
        
    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    def get_initial_latent_dists(self, B, K):
        mean = torch.zeros((B, K, self.latent_state_size), device=ptu.device)
        scale = torch.ones_like(mean)
        latent_dist_0 = NormalLatentStateDistribution(mean, scale, do_build_rsample=False)
        return latent_dist_0

    def forward(self, latent_dist_sample_tm1, action_tm1, agent_pose_t=None):
        """

        :param latent_dist_sample_tm1: (B, K, S)
        :param action_tm1: (B, A)
        :returns: 
        :rtype: 

        """
        assert(action_tm1.ndim == 2)
        K = latent_dist_sample_tm1.shape[-2]
        action_tm1_tile = K_tile(K, action_tm1)
        action_tm1_embed = self.fc_embed_action(action_tm1_tile)
        latent_dist_params_preact_t = self.fc_latent_dist_params(torch.cat((latent_dist_sample_tm1, action_tm1_embed), axis=-1))
        dist_means, _dist_std_dev = torch.chunk(latent_dist_params_preact_t, 2, dim=-1)
        dist_means = torch.clamp(dist_means, -1 * self._abs_mean_range, self._abs_mean_range)
        dist_stddevs = torch.max(self.std_act(_dist_std_dev), self.min_std_dev)
        prior_latent_dist_t = NormalLatentStateDistribution(dist_means, dist_stddevs, do_build_rsample=False)
        self.most_recent_dist = prior_latent_dist_t
        return prior_latent_dist_t

class RNNNormalPriorModel(PriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, min_std_dev=1e-2, rnn_state_size=600, hidden_sizes=[200], std_act='softplus', embed_action=False, abs_mean_range=1e3, **kwargs):
        super().__init__()
        self._abs_mean_range = abs_mean_range
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = 2 * self.latent_state_size
        self._rnn_state_size = rnn_state_size
        self._action_size = action_size

        self.embed_action = embed_action
        if self.embed_action:
            self.fc_embed_action = nn.Linear(self.action_size, self.latent_state_size)
            self._input_size = self.latent_state_size + self.fc_embed_action.out_features
        else:
            self.fc_embed_action = lambda x: x
            self._input_size = self.latent_state_size + self.action_size

        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        # Embed -> RNN -> predict dist params
        self.fc_embed_state_and_action = nn.Linear(self._input_size, self.latent_state_size)
        self.rnn = nn.GRUCell(self.latent_state_size, self.latent_state_size)
        self.fc_latent_dist_params = build_mlp(self.latent_state_size, hidden_sizes, self.latent_distribution_size)
        self.std_act = get_act(std_act)

    @property
    def rollout_state(self):
        return self._rollout_state

    @rollout_state.setter
    def rollout_state(self, x):
        self._rollout_state = x
        
    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    def get_initial_latent_dists(self, B, K):
        mean = torch.zeros((B, K, self.latent_state_size), device=ptu.device)
        scale = torch.ones_like(mean)
        latent_dist_0 = NormalLatentStateDistribution(mean, scale, do_build_rsample=False)
        rnn_state = torch.zeros((B*K, self.latent_state_size), device=ptu.device)
        self.rollout_state = dict(rnn_state=rnn_state)
        return latent_dist_0

    def forward(self, latent_dist_sample_tm1, action_tm1, agent_pose_t=None):
        """

        :param latent_dist_sample_tm1: (B, K, S)
        :param action_tm1: (B, A)
        :returns: 
        :rtype: 

        """
        assert(action_tm1.ndim == 2)
        B, K, D = latent_dist_sample_tm1.shape
        action_tm1_tile = K_tile(K, action_tm1)
        embedding_t = self.fc_embed_state_and_action(torch.cat((latent_dist_sample_tm1, self.fc_embed_action(action_tm1_tile)), axis=-1))
        rnn_state_t = self.rnn(embedding_t.view((B*K, D)), self.rollout_state['rnn_state'])
        latent_dist_params_preact_t = self.fc_latent_dist_params(rnn_state_t)
        dist_means, _dist_std_dev = torch.chunk(latent_dist_params_preact_t, 2, dim=-1)
        dist_means = torch.clamp(dist_means, -1 * self._abs_mean_range, self._abs_mean_range)
        dist_stddevs = torch.max(self.std_act(_dist_std_dev), self.min_std_dev)
        prior_latent_dist_t = NormalLatentStateDistribution(dist_means.view((B, K, D)), dist_stddevs.view((B, K, D)), do_build_rsample=False)
        self.rollout_state = DetachableDict(rnn_state=rnn_state_t)
        self.most_recent_dist = prior_latent_dist_t
        return prior_latent_dist_t

class MLPSingle3DParamPriorModel(PriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, dist_class, dist_kwargs={}, min_std_dev=1e-2, hidden_sizes=[200], **kwargs):
        super().__init__()
        self._dist_class = dist_class
        self._dist_kwargs = dist_kwargs
        if not np.isclose(np.sqrt(latent_state_size) % 1, 0):
            raise ValueError(f"latent_state_size={latent_state_size} is not a perfect square!")
        self._sqrt_latent_state_size = int(round(np.sqrt(latent_state_size)))
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = self.latent_state_size
        self._action_size = action_size
        self._event_shape = (self._sqrt_latent_state_size, self._sqrt_latent_state_size)
        self._event_size = (self.latent_state_size,)

        self._input_size = self.latent_state_size + self.action_size
        
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.fc_latent_dist_params = build_mlp(self._input_size, hidden_sizes, self.latent_distribution_size)
        self.activation_function = F.elu
        log.info(f"Initialized prior: {self.__class__.__name__}")

    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    @property
    def rollout_state(self): return DetachableDict()
        
    @rollout_state.setter
    def rollout_state(self, x): pass

    def get_initial_latent_dists(self, B, K):
        logits = torch.zeros((B, K, self._sqrt_latent_state_size, self._sqrt_latent_state_size), device=ptu.device)
        latent_dist_0 = self._dist_class(logits, **self._dist_kwargs)
        self._batch_shape = (B, K)
        return latent_dist_0        
        
    def forward(self, latent_dist_sample_tm1, action_tm1, agent_pose_t=None):
        latent_dist_sample_tm1 = latent_dist_sample_tm1.view(self._batch_shape + self._event_size)
        
        action_tm1_tile = K_tile(self._batch_shape[-1], action_tm1)
        mlp_input = torch.cat((latent_dist_sample_tm1, action_tm1_tile), axis=-1)
        latent_dist_params = self.fc_latent_dist_params(mlp_input).view(self._batch_shape + self._event_shape)        
        prior_latent_dist_t = self._dist_class(latent_dist_params, **self._dist_kwargs)
        return prior_latent_dist_t


class RNNSingle3DParamPriorModel(PriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, dist_class, dist_kwargs={}, min_std_dev=1e-2, hidden_sizes=[200], embed_action=True, input_previous_samples=False, **kwargs):
        super().__init__()
        self._dist_class = dist_class
        self._dist_kwargs = dist_kwargs
        if not np.isclose(np.sqrt(latent_state_size) % 1, 0):
            raise ValueError(f"latent_state_size={latent_state_size} is not a perfect square!")
        self._sqrt_latent_state_size = int(round(np.sqrt(latent_state_size)))
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = self.latent_state_size
        self._event_shape = (self._sqrt_latent_state_size, self._sqrt_latent_state_size)
        self._action_size = action_size
        self._event_size = (self.latent_state_size,)

        self.fc_embed_action = nn.Linear(self.action_size, self.latent_state_size)
        
        self.is_belief = False
        self._input_size = self.latent_state_size + self.fc_embed_action.out_features
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.rnn = nn.GRUCell(self._input_size, self.latent_state_size)
        self.fc_latent_dist_params = nn.Linear(self.latent_state_size, self.latent_distribution_size)
        log.info(f"Initialized prior: {self.__class__.__name__}")

    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    @property
    def rollout_state(self):
        return self._rollout_state

    @rollout_state.setter
    def rollout_state(self, v):
        self._rollout_state = v

    def get_initial_latent_dists(self, B, K):
        logits = torch.zeros((B, K, self._sqrt_latent_state_size, self._sqrt_latent_state_size), device=ptu.device)
        rnn_state = torch.zeros((B*K, self._latent_state_size), device=ptu.device)
        latent_dist_0 = self._dist_class(logits, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)
        self._batch_shape = (B, K)
        self._batch_size = (B*K,)
        self._rollout_state = DetachableDict(rnn_state=rnn_state)
        return latent_dist_0        
        
    def forward(self, latent_dist_sample_tm1, action_tm1, agent_pose_t=None):
        action_tm1 = K_tile(self._batch_shape[-1], action_tm1)
        action_tm1 = K_tile(self._batch_shape[-1], action_tm1)
        rnn_input_t = torch.cat((self.fc_embed_action(action_tm1), latent_dist_sample_tm1.view(self._batch_shape + self._event_size)), -1)
        rnn_input_2d = rnn_input_t.view(self._batch_size + (self._input_size,))
        rnn_state_tp1 = self.rnn(rnn_input_2d, self.rollout_state['rnn_state'])
        latent_dist_params = self.fc_latent_dist_params(rnn_state_tp1).view(self._batch_shape + self._event_shape)
        prior_latent_dist_t = self._dist_class(latent_dist_params, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)
        self._rollout_state = DetachableDict(rnn_state=rnn_state_tp1)
        return prior_latent_dist_t

MAX_STD_DEV = 1e2

#------------------ POSTERIORS -------------------------
class MLPNormalPosteriorModel(PosteriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, observation_size, min_std_dev=1e-2, hidden_sizes=[200], std_act='softplus', **kwargs):
        super().__init__()
        self._latent_state_size = latent_state_size
        self._observation_size = observation_size
        self._action_size = action_size
        self._latent_distribution_size = 2 * self.latent_state_size
        self._input_size = self.observation_size
        assert(min_std_dev <= 0.1)
        self.min_std_dev_v = min_std_dev
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.fc_latent_dist_params = build_mlp(self._input_size, hidden_sizes, self.latent_distribution_size)
        self.std_act = get_act(std_act)
        log.info(f"Initialized posterior: {self.__class__.__name__}")
        self.is_belief = True

    @property
    def rollout_state(self): return DetachableDict()
    
    @rollout_state.setter
    def rollout_state(self, x): pass
        
    @property
    def observation_size(self): return self._observation_size

    @property
    def action_size(self): return self._action_size    

    @property
    def latent_state_size(self): return self._latent_state_size

    @property
    def latent_distribution_size(self): return self._latent_distribution_size

    def get_initial_latent_dists(self, B, K):
        mean = torch.zeros((B, K, self.latent_state_size), device=ptu.device)
        scale = torch.ones_like(mean)
        latent_dist_0 = NormalLatentStateDistribution(mean, scale, do_build_rsample=True, is_belief=self.is_belief)
        return latent_dist_0

    def forward(self, latent_dist_sample_tm1, observation_t, action_tm1, prior_latent_dist_t, prior_rollout_state_t, agent_pose_t=None):
        assert(action_tm1.ndim == 2)
        K = latent_dist_sample_tm1.shape[-2]
        action_tm1_tile, observation_t_tile = K_tile(K, action_tm1, observation_t)
        latent_dist_params_preact = self.fc_latent_dist_params(observation_t_tile)
        dist_means, _dist_std_dev = torch.chunk(latent_dist_params_preact, 2, dim=-1)
        dist_stddevs = torch.max(self.std_act(_dist_std_dev), self.min_std_dev)

        # Relative means and stddevs.
        posterior_mean = prior_latent_dist_t.state_mean + dist_means
        posterior_stddev = torch.clamp(prior_latent_dist_t.state_stddev * dist_stddevs, min=self.min_std_dev_v, max=MAX_STD_DEV)
        posterior_latent_dist_t = NormalLatentStateDistribution(posterior_mean, posterior_stddev, do_build_rsample=True, is_belief=self.is_belief)
        return posterior_latent_dist_t

class RNNNormalPosteriorModel(PosteriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, observation_size, min_std_dev=1e-2, hidden_sizes=[200], rnn_state_size=600, std_act='softplus', embed_action=False, input_previous_samples=False, **kwargs):
        super().__init__()
        self._latent_state_size = latent_state_size
        self._observation_size = observation_size
        self._action_size = action_size
        self._latent_distribution_size = 2 * self.latent_state_size
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)

        if embed_action:
            self.fc_embed_action = nn.Linear(self.action_size, self.latent_state_size)
            self._input_size = self.observation_size + self.fc_embed_action.out_features
        else:
            self.fc_embed_action = lambda x: x
            self._input_size = self.observation_size + self.action_size

        self.input_previous_samples = input_previous_samples
        
        if self.input_previous_samples:
            self._input_size += self.latent_state_size

        self.rnn_state_size = rnn_state_size
        self.fc_embed_belief_observation_and_action = build_mlp(self._input_size, hidden_sizes, self.latent_state_size)
        self.rnn = nn.GRUCell(self.latent_state_size, self.rnn_state_size)
        self.fc_latent_dist_params = nn.Linear(self.rnn_state_size, self.latent_distribution_size)
        self.activation_function = F.elu
        # If we use previous samples, it's not a belief distribution.
        self.is_belief = not self.input_previous_samples
        self.std_act = get_act(std_act)
        log.info(f"Initialized posterior: {self.__class__.__name__}")
        
    @property
    def observation_size(self): return self._observation_size

    @property
    def action_size(self): return self._action_size    

    @property
    def latent_state_size(self): return self._latent_state_size

    @property
    def latent_distribution_size(self): return self._latent_distribution_size

    @property
    def rollout_state(self):
        return self._rollout_state

    @rollout_state.setter
    def rollout_state(self, x): self._rollout_state = x

    def get_initial_latent_dists(self, B, K):
        mean = torch.zeros((B, K, self.latent_state_size), device=ptu.device)
        scale = torch.ones_like(mean)
        rnn_state = torch.zeros((B*K, self.rnn_state_size), device=ptu.device)
        latent_dist_0 = NormalLatentStateDistribution(mean, scale, do_build_rsample=True, is_belief=self.is_belief)
        self._rollout_state = DetachableDict(rnn_state=rnn_state)
        return latent_dist_0

    def forward(self, latent_dist_sample_tm1, observation_t, action_tm1, prior_latent_dist_t, prior_rollout_state_t, agent_pose_t=None):
        B, K, D = latent_dist_sample_tm1.shape

        action_tm1_tile, observation_t_tile = K_tile(K, action_tm1, observation_t)

        if self.input_previous_samples:
            embedding_t = self.fc_embed_belief_observation_and_action(torch.cat((observation_t_tile, self.fc_embed_action(action_tm1_tile), latent_dist_sample_tm1), axis=-1))
        else:
            # NB here we're not using any previous latents.
            del latent_dist_sample_tm1
            embedding_t = self.fc_embed_belief_observation_and_action(torch.cat((observation_t_tile, self.fc_embed_action(action_tm1_tile)), axis=-1))
            
        rnn_state_tp1 = self.rnn(embedding_t.view((B*K, D)), self._rollout_state['rnn_state'])
        latent_dist_params_preact = self.fc_latent_dist_params(rnn_state_tp1)
        dist_means, _dist_std_dev = torch.chunk(latent_dist_params_preact, 2, dim=-1)
        # Make a randomly-initialized network have small std dev predictions.
        dist_stddevs = torch.max(self.std_act(_dist_std_dev), self.min_std_dev)
        posterior_latent_dist_t = NormalLatentStateDistribution(dist_means.view((B, K, D)), dist_stddevs.view((B, K, D)), do_build_rsample=True, is_belief=self.is_belief)
        # Store RNN state alongside the belief state.
        self._rollout_state = DetachableDict(rnn_state=rnn_state_tp1)
        return posterior_latent_dist_t

def K_tile(K, *args):
    result = []
    for arg in args:
        if arg.ndim == 2:
            result.append(arg[:, None].repeat((1, K, 1)))
        elif arg.ndim == 3:
            assert(arg.shape[1] == K)
            result.append(arg)
        elif arg.ndim == 4:
            result.append(arg[:, None].repeat((1, K, 1, 1, 1)))
        else:
            raise ValueError(arg.ndim)
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)

class MLPSingle3DParamPosteriorModel(PosteriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, observation_size, dist_class, dist_kwargs={}, min_std_dev=1e-2, hidden_sizes=[200], embed_action=True, input_previous_samples=False, **kwargs):
        super().__init__()
        self._dist_class = dist_class
        self._dist_kwargs = dist_kwargs
        if not np.isclose(np.sqrt(latent_state_size) % 1, 0):
            raise ValueError(f"latent_state_size={latent_state_size} is not a perfect square!")
        self._sqrt_latent_state_size = int(round(np.sqrt(latent_state_size)))
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = self.latent_state_size
        self._event_shape = (self._sqrt_latent_state_size, self._sqrt_latent_state_size)
        self._action_size = action_size
        self._observation_size = observation_size        
        self._event_size = (self.latent_state_size,)
        self._input_size = self.observation_size 

        # This is properly not a belief distribution, since it depends on previous latents. Setting this causes weighted-averaging over the particles to occur when the beliefs are queried.
        self.is_belief = False
        
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.fc_latent_dist_params = build_mlp(self._input_size, hidden_sizes, self.latent_state_size)
        log.info(f"Initialized prior: {self.__class__.__name__}")

    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    @property
    def observation_size(self): return self._observation_size

    @property
    def rollout_state(self): return DetachableDict()

    @rollout_state.setter
    def rollout_state(self, v): pass

    def get_initial_latent_dists(self, B, K):
        logits = torch.zeros((B, K, self._sqrt_latent_state_size, self._sqrt_latent_state_size), device=ptu.device)
        latent_dist_0 = self._dist_class(logits, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)
        self._batch_shape = (B, K)
        self._batch_size = (B*K,)
        return latent_dist_0        
        
    def forward(self, latent_dist_sample_tm1, observation_t, action_tm1, prior_latent_dist_t, prior_rollout_state_t, agent_pose_t=None):
        if self._observe_agent_pose: assert(agent_pose_t is not None)

        if self._observe_agent_pose:
            assert(agent_pose_t is not None)
            action_tm1, observation_t, agent_pose_t_tile = K_tile(self._batch_shape[-1], action_tm1, observation_t, agent_pose_t)
            fc_input = torch.cat((observation_t, agent_pose_t_tile), axis=-1)
        else:
            action_tm1, observation_t = K_tile(self._batch_shape[-1], action_tm1, observation_t)
            fc_input = observation_t
            
        latent_dist_params = self.fc_latent_dist_params(fc_input)
        # Relative dist params to priors
        posterior_latent_dist_params = latent_dist_params.view(self._batch_shape + self._event_shape) + prior_latent_dist_t.state_logits
        posterior_latent_dist_t = self._dist_class(posterior_latent_dist_params, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)        
        return posterior_latent_dist_t

class RNNSingle3DParamPosteriorModel(PosteriorModel, torch.nn.Module):
    def __init__(self, latent_state_size, action_size, observation_size, dist_class, dist_kwargs={}, min_std_dev=1e-2, hidden_sizes=[200], embed_action=True, input_previous_samples=False, **kwargs):
        super().__init__()
        self._dist_class = dist_class
        self._dist_kwargs = dist_kwargs
        if not np.isclose(np.sqrt(latent_state_size) % 1, 0):
            raise ValueError(f"latent_state_size={latent_state_size} is not a perfect square!")
        self._sqrt_latent_state_size = int(round(np.sqrt(latent_state_size)))
        self._latent_state_size = latent_state_size
        self._latent_distribution_size = self.latent_state_size
        self._event_shape = (self._sqrt_latent_state_size, self._sqrt_latent_state_size)
        self._action_size = action_size
        self._observation_size = observation_size        
        self._event_size = (self.latent_state_size,)

        if embed_action:
            self.fc_embed_action = nn.Linear(self.action_size, self.latent_state_size)
            self._input_size = self.observation_size + self.fc_embed_action.out_features
        else:
            self.fc_embed_action = lambda x: x
            self._input_size = self.observation_size + self.action_size

        self.input_previous_samples = input_previous_samples
        if self.input_previous_samples: self._input_size += self.latent_state_size
        self.is_belief = not self.input_previous_samples
        
        assert(min_std_dev <= 0.1)
        self.min_std_dev = torch.full(size=(1,), fill_value=min_std_dev, device=ptu.device)
        self.fc_embed_observation_and_action = build_mlp(self._input_size, hidden_sizes, self.latent_state_size)
        self.rnn = nn.GRUCell(self.latent_state_size, self.latent_state_size)
        self.fc_latent_dist_params = nn.Linear(self.latent_state_size, self.latent_distribution_size)
        log.info(f"Initialized prior: {self.__class__.__name__}")

    @property
    def latent_state_size(self): return self._latent_state_size
        
    @property
    def latent_distribution_size(self): return self._latent_distribution_size
    
    @property
    def action_size(self): return self._action_size

    @property
    def observation_size(self): return self._observation_size

    @property
    def rollout_state(self):
        return self._rollout_state

    @rollout_state.setter
    def rollout_state(self, v):
        self._rollout_state = v

    def get_initial_latent_dists(self, B, K):
        logits = torch.zeros((B, K, self._sqrt_latent_state_size, self._sqrt_latent_state_size), device=ptu.device)
        rnn_state = torch.zeros((B*K, self._latent_state_size), device=ptu.device)
        latent_dist_0 = self._dist_class(logits, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)
        self._batch_shape = (B, K)
        self._batch_size = (B*K,)
        self._rollout_state = DetachableDict(rnn_state=rnn_state)
        return latent_dist_0        
        
    def forward(self, latent_dist_sample_tm1, observation_t, action_tm1, prior_latent_dist_t, prior_rollout_state_t, agent_pose_t=None):
        action_tm1, observation_t = K_tile(self._batch_shape[-1], action_tm1, observation_t)        
        rnn_input_t = self.fc_embed_observation_and_action(torch.cat((self.fc_embed_action(action_tm1), observation_t), axis=-1))

        if self.input_previous_samples:
            rnn_input_t = self.fc_embed_observation_and_action(torch.cat((self.fc_embed_action(action_tm1), observation_t, latent_dist_sample_tm1.view(self._batch_shape + self._event_size)), axis=-1))
        else:
            # NB here we're not using any previous latents.
            del latent_dist_sample_tm1
            rnn_input_t = self.fc_embed_observation_and_action(torch.cat((self.fc_embed_action(action_tm1), observation_t), axis=-1))
        
        rnn_input_2d = rnn_input_t.view(self._batch_size + self._event_size)
        rnn_state_tp1 = self.rnn(rnn_input_2d, self.rollout_state['rnn_state'])
        latent_dist_params = self.fc_latent_dist_params(rnn_state_tp1).view(self._batch_shape + self._event_shape)
        posterior_latent_dist_t = self._dist_class(latent_dist_params, do_build_rsample=True, is_belief=self.is_belief, **self._dist_kwargs)
        self._rollout_state = DetachableDict(rnn_state=rnn_state_tp1)
        return posterior_latent_dist_t

def get_class_and_kwargs_of_prior(prior_model_kwargs):
    prior_model_kwargs = copy.deepcopy(prior_model_kwargs)
    mtype = prior_model_kwargs.pop('type')
    if mtype == 'mlp_normal':
        Class, kwargs = MLPNormalPriorModel, prior_model_kwargs
    elif mtype in ('mlp_categorical_st', 'mlp_independent_categorical_st'):
        prior_model_kwargs['dist_class'] = IndependentCategoricalLatentStateDistribution
        prior_model_kwargs['dist_kwargs'] = dict(mode='st')
        Class, kwargs = MLPSingle3DParamPriorModel, prior_model_kwargs
    elif mtype in ('rnn_categorical_st', 'rnn_independent_categorical_st'):
        prior_model_kwargs['dist_class'] = IndependentCategoricalLatentStateDistribution
        prior_model_kwargs['dist_kwargs'] = dict(mode='st')
        Class, kwargs = RNNSingle3DParamPriorModel, prior_model_kwargs
    elif mtype == 'rnn_normal':
        Class, kwargs = RNNNormalPriorModel, prior_model_kwargs
    else:
        raise NotImplementedError(mtype)
    return Class, kwargs    

def get_class_and_kwargs_of_posterior(posterior_model_kwargs):
    posterior_model_kwargs = copy.deepcopy(posterior_model_kwargs)
    mtype = posterior_model_kwargs.pop('type')
    
    if mtype == 'rnn_normal':
        Class, kwargs = RNNNormalPosteriorModel, posterior_model_kwargs
    elif mtype == 'mlp_normal':
        Class, kwargs = MLPNormalPosteriorModel, posterior_model_kwargs
    elif mtype in ('rnn_categorical_st', 'rnn_independent_categorical_st'):
        posterior_model_kwargs['dist_class'] = IndependentCategoricalLatentStateDistribution
        posterior_model_kwargs['dist_kwargs'] = dict(mode='st')
        Class, kwargs = RNNSingle3DParamPosteriorModel, posterior_model_kwargs
    elif mtype in ('mlp_categorical_st', 'mlp_independent_categorical_st'):
        posterior_model_kwargs['dist_class'] = IndependentCategoricalLatentStateDistribution
        posterior_model_kwargs['dist_kwargs'] = dict(mode='st')
        Class, kwargs = MLPSingle3DParamPosteriorModel, posterior_model_kwargs        
    else:
        raise NotImplementedError(mtype)
    return Class, kwargs

class VBFModelTrainer:
    @classu.member_initialize
    def __init__(self, vbf_model, optimizer_klass=optim.RAdam, learning_rate=1e-3, grad_clip_norm=1000, optimizer_config={}, parallelized=False,
                 **kwargs):
        self.params_list = list(self.vbf_model.parameters())
        self.optimizer = optimizer_klass(self.params_list, lr=self.learning_rate)
        self.t = 0            

        def get_scheduler(name):
            """For example, if model_optimizer.loss_mode_schedule=less_than_threshold, then, get_scheduler("loss_mode"):
               
                      ns = "loss_mode_schedule"
                      schedule_name = "less_than_threshold" 
                      schedule_name_title = "LessThanThreshold"
                      klass = mlu.LessThanThresholdSchedule
                      inst = mlu.LessThanThresholdSchedule(**optimizer_config["loss_mode_schedule_kwargs"]["less_than_threshold"])

                Or if model_optimizer.beta_schedule==constant, then get_scheduler("beta"):
                      inst = mlu.ConstantSchedule(**optimizer_config["beta_schedule_kwargs"]["constant"])
            """
            ns = name + '_schedule'
            schedule_name = optimizer_config[ns]
            schedule_name_title = ''.join([_.title() for _ in schedule_name.split('_')])
            klass = getattr(mlu, schedule_name_title + 'Schedule')
            inst = klass(**optimizer_config[ns + '_kwargs'][schedule_name])
            return inst

        self.alpha_scheduler = get_scheduler('alpha')
        self.beta_scheduler = get_scheduler('beta')
        self.loss_mode_scheduler = get_scheduler('loss_mode')
        log.info(f"Instantiated {self}")
        log.debug(f"Instantiated {self}")

    def save(self, step):
        model_basename = f"model_{step:08d}.pt"
        # Save model.
        torch.save(self.vbf_model.state_dict(), model_basename)
        # Save optimizer.
        torch.save(self.optimizer.state_dict(), "optimizer_of_" + model_basename)
        return model_basename

    def load(self, filename):
        # Load model.
        self.vbf_model.load_state_dict(torch.load(filename, map_location=ptu.device))

        # Try to load optimizer.
        try:
            head, tail = os.path.split(filename)
            opt_fn = f"{head}/optimizer_of_{tail}"
            log.info(f"Loading optimizer filename: '{opt_fn}'")
            self.optimizer.load_state_dict(torch.load(opt_fn, map_location=ptu.device))
            # Override learning rate with new learning rate
            for pg in self.optimizer.param_groups: pg['lr'] = self.learning_rate
        except FileNotFoundError as e:
            log.error(f"Couldn't load model optimizer! Error: '{e}'")

    def __repr__(self):
        return (f"{self.__class__.__name__}(learning_rate={self.learning_rate:.2g}, beta={self.beta_scheduler}, " +
                f"alpha={self.alpha_scheduler}, loss_mode={self.loss_mode_scheduler}, " +
                f"grad_clip_norm={self.grad_clip_norm}, opt={self.optimizer})")

    def fit_minibatch(self, batch):
        log.debug(f"In {inspect.currentframe().f_code.co_name} of {self.__class__.__name__} instance.")
        # Set the model to be in training mode.
        self.vbf_model.train()
        self.optimizer.zero_grad()
        beta = self.beta_scheduler.get_value(self.t)
        alpha = self.alpha_scheduler.get_value(self.t)
        loss_mode = self.loss_mode_scheduler.get_value(self.t)

        compute_losses_online_and_truncate_BPTT = self.vbf_model.model_kwargs['loss']['compute_online']

        def optimizer_step_and_reset():
            preclip_norm = nn.utils.clip_grad_norm_(self.params_list, self.grad_clip_norm, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        with Timer('forward and losses'):
            losses_and_stats = self.vbf_model.belief_filter_and_compute_losses(
                batch=batch,
                with_prior_observation_dists=False,
                t=self.t,
                compute_losses_online_and_truncate_BPTT=compute_losses_online_and_truncate_BPTT,
                beta=beta,
                alpha=alpha,
                loss_mode=loss_mode,
                include_diagnostics=False,
                optimizer_step_and_reset_func=optimizer_step_and_reset)

        if not compute_losses_online_and_truncate_BPTT:
            with Timer('backward'):
                losses_and_stats['total_loss'].backward()

            log.debug(f"Loss: {losses_and_stats['total_loss']:.3g}")
            preclip_norm = nn.utils.clip_grad_norm_(self.params_list, self.grad_clip_norm, norm_type=2)
            losses_and_stats['grad_norm'] = preclip_norm
            losses_and_stats['t'] = self.t
            if preclip_norm > 1e2 * self.grad_clip_norm:
                log.warning(f"Gradient pre-clip norm is very large: {preclip_norm:.2g}!")

            with Timer('step'):
                self.optimizer.step()
                
        self.t += 1
        # Update the schedulers with the losses and diagnostics.
        self.beta_scheduler.update(losses_and_stats)
        self.alpha_scheduler.update(losses_and_stats)
        self.loss_mode_scheduler.update(losses_and_stats)

        losses_and_stats['beta'] = beta
        losses_and_stats['alpha'] = alpha
        return ptu.to_numpy(losses_and_stats)

    def randomized_evaluation(self, *args, **kwargs):
        # Always evaluate with standard parameters.
        beta = 1.0
        alpha = 0.5
        loss_mode = 'regular'
        self.vbf_model.eval()
        return self.vbf_model.randomized_evaluation(*args, **kwargs, beta=beta, alpha=alpha, loss_mode=loss_mode)

class BayesianVBFModel:
    def __init__(self, vbf_model, swag_cfg, ensemble_cfg):
        self.vbf_model = vbf_model
        self.swag_cfg = swag_cfg

        if self.swag_cfg["model_key"] == 'prior_model':
            pass
        else:
            raise NotImplementedError(f"Model component={self.swag_cfg['model_key']} not implemented yet") 
        
        from swag.posteriors import SWAG
        # Assumes the VBF model provides the model_key model and its init_kwargs
        self.model_kwargs = getattr(vbf_model, f'_{self.swag_cfg["model_key"]}_init_kwargs')
        self.model_class = getattr(vbf_model, f'_{self.swag_cfg["model_key"]}_class')
        self.swag_model_getter = lambda: getattr(vbf_model, f'_{self.swag_cfg["model_key"]}')

        # SWAG will instantiate the model_class and store statistics alongside the parameters therein.
        # The more models used (max_num_models), the more samples we have to approximate the posterior.
        self.swag_posterior = SWAG(base=self.model_class, no_cov_mat=swag_cfg['no_cov_mat'], max_num_models=swag_cfg['max_num_models'], **self.model_kwargs)
        self.swag_posterior.to(ptu.device)
        self.ensemble = self._create_ensemble(ensemble_cfg.N)

    def _create_ensemble(self, N):
        # Sample model parameters from the bayesian posterior and compute the predictions
        ensemble = []
        for _ in range(N):
            # Get the models from the main model so that we don't have to allocate new parameters for the things we're not being Bayesian about.
            models = self.vbf_model.get_models()
            # Remove the model we'll create a Bayesian posterior for, which results in the instance allocating a new nn.Module for it.
            models.pop(self.swag_cfg["model_key"])
            kwargs = {**self.vbf_model._init_kwargs.copy(), 'models': models}
            ensemble.append(VBFModel(**kwargs))
        return ensemble

    def resample_ensemble_parameters(self):
        model_key = f'_{self.swag_cfg["model_key"]}'
        for ensemble_model in self.ensemble:
            # Sampling from the posterior sets the parameters of the internal ("base") member of the SWAG posterior
            self.swag_posterior.sample(block=True, cov=False)
            # Get the component of the VBFModel that we'll change the parameters of.
            ensemble_model_component = getattr(ensemble_model, model_key)
            # Extract out the full SWAG state.
            source_component_state = self.swag_posterior.base.state_dict()
            source_component_state_pruned = {k: v for k, v in source_component_state.items() if (k.rfind('mean') == -1 and k.rfind('sq_mean') == -1 and k.rfind('cov_mat_sqrt') == -1)}
            # Set the model component's state to be the sampled components state
            ensemble_model_component.load_state_dict(source_component_state_pruned)

    def load(self, fn):
        head, tail = os.path.split(fn)
        swag_fn = f"{head}/swag_posterior_of_{tail}"
        log.info(f"Loading SWAG posterior: {swag_fn}")
        self.swag_posterior.load_state_dict(torch.load(swag_fn, map_location=ptu.device))

    def save(self, fn):
        head, tail = os.path.split(fn)
        swag_fn = f"{head}/swag_posterior_of_{tail}"
        torch.save(self.swag_posterior.state_dict(), swag_fn)
        
class VBFReplayBufferTrainer:
    @classu.member_initialize
    def __init__(self, replay_buffer, vbf_model_trainer, B, H,
                 bayesian_vbf_model=None,
                 swa_cfg=dict(swa_freq=500, swa_start=1000)):

        self._epochs = 0
        self._minibatches = 0
        self.hist_keys = {'particle_weights_t'}
        self._sequential_yielder = None

    def save(self, *args, **kwargs):
        self.vbf_model_trainer.save(*args, **kwargs)

    def visualize(self):
        return self.vbf_model_trainer.vbf_model.replay_buffer_vis(self.replay_buffer, self.H)
        
    def fit_epochs(self, writer, number, visualize=False, mode='pretrain'):
        if number > 1:
            for i in tqdm.trange(int(round(number))):
                res = self.fit_epoch(writer)
                # Save the model once per epoch
                self.save(writer.step)
                if visualize: writer.add_image(f'{mode}_partial_rollout', self.visualize(), writer.step, dataformats='HWC')
            return res
        elif number > 0:
            res = self.fit_partial_epoch(writer, number)
            # Assume saving is managed outside.
            # self.save(writer.step)
            return res
        elif number == 0:
            return None
        else:
            raise ValueError(number)        

    def fit_epoch(self, writer):
        return self.fit_partial_epoch(writer, fraction=1.)

    def fit_partial_epoch(self, writer, fraction=0.5):
        assert(0 < fraction <= 1.)
        n_approx_windows = self.replay_buffer.get_approximate_minibatch_count(batch_size=self.B)
        n_fraction_windows = max(int(round(n_approx_windows * fraction)), 1)
        metadata = dict(epochs=self._epochs)

        if self.vbf_model_trainer.parallelized:
            self._sequential_yielder = self.replay_buffer.sequential_batch_windows(self.B, window_len=self.H, obs_key=self.vbf_model_trainer.vbf_model.observation_key)
        
        for i in tqdm.trange(n_fraction_windows):
            try:
                ret = self._fit_minibatch()
                ret.update(metadata)

                swag_has_begun = self.bayesian_vbf_model is not None and self._minibatches >= self.swa_cfg['start']
                swag_now = swag_has_begun and (self._minibatches % self.swa_cfg['freq'] == 0)
                # Incorporate another parameter estimate into the posterior.
                if swag_now:
                    self.bayesian_vbf_model.swag_posterior.collect_model(self.bayesian_vbf_model.swag_model_getter())

                for fl in ret.items():
                    if fl[0] in self.hist_keys:
                        writer.add_histogram(f'VBF_opt/{fl[0]}', fl[1], writer.step)
                    else:
                        writer.add_scalar(f'VBF_opt/{fl[0]}', fl[1], writer.step)
                writer.increment()
            except StopIteration:
                log.info(f"Stopping iteration at minibatch {i} (reached end of epoch)")
                break
        self._epochs += fraction
        self._sequential_yielder = None
        return ret
    
    def _fit_minibatch(self):
        if self.vbf_model_trainer.parallelized:
            assert(self._sequential_yielder is not None)
            train_data, is_start = next(self._sequential_yielder)
            train_data = ptu.np_to_pytorch_batch(train_data)
            train_data['is_start'] = is_start
        else:
            train_data = self.replay_buffer.random_batch_window(batch_size=self.B, window_len=self.H, obs_key=self.vbf_model_trainer.vbf_model.observation_key)
            train_data = ptu.np_to_pytorch_batch(train_data)
        # Convert numpy arrays to torch.Tensors on the default device.
        losses_and_stats = self.vbf_model_trainer.fit_minibatch(train_data)
        self._minibatches += 1
        return losses_and_stats

    def infer_single(self):
        train_data = self.replay_buffer.random_batch_window(batch_size=1, window_len=self.H, obs_key=self.vbf_model_trainer.vbf_model.observation_key)
        # Convert numpy arrays to torch.Tensors on the default device.
        train_data = ptu.np_to_pytorch_batch(train_data)
        
        bfs = self.vbf_model_trainer.vbf_model.belief_filter(batch=train_data, with_prior_observation_dists=True, with_posterior_observation_dists=True)
        return bfs

class MeanSamplingNormal(Normal):
    # Override the sampling procedure of a Normal distribution to just return the mean.
    def sample(self, sample_shape):
        return self.mean.detach().expand(self._extended_shape(sample_shape))
    
    def rsample(self, sample_shape):
        return self.mean.expand(self._extended_shape(sample_shape))

class SliceableIndependentOfNormal(Independent):
    def slice0(self, slice_):
        loc = self.loc[slice_]
        scale = self.scale[slice_]
        return Independent(self.base_distribution.__class__(loc=loc, scale=scale), reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

class VisualObservationModel(torch.nn.Module):
    __constants__ = ['embedding_size']
    
    def __init__(self, state_size, embedding_size, activation_function='elu', target_output_size=64, learnable_stddev=False, mixture=False, k_modes=5, continuous_bernoulli=False,
                 state_partition_index=None, k_steps=1, no_partition=False, observe_agent_pose=False, agent_pose_size=2):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.k_steps = k_steps
        self.embedding_size = embedding_size
        self.learnable_stddev = learnable_stddev
        self.mixture = mixture
        self.k_modes = k_modes
        if k_steps > 1 and not no_partition:
            assert(state_size % k_steps == 0)
            state_partition_index = state_size // k_steps
        elif state_partition_index in (None, 'None') or no_partition:
            state_partition_index = state_size
        else:
            raise ValueError

        self.state_partition_index = state_partition_index
        self.agent_pose_size = agent_pose_size
        self.observe_agent_pose = observe_agent_pose
        
        if self.observe_agent_pose:
            self.state_partition_index += self.agent_pose_size
        self.fc1 = nn.Linear(self.state_partition_index, embedding_size)
        
        self.continuous_bernoulli = continuous_bernoulli

        self.min_std_dev = 1e-5                    

        if self.learnable_stddev and not continuous_bernoulli:
            self.log_stddev = torch.nn.Parameter(torch.full(size=(1,), fill_value=np.log(1-1e-5), device=ptu.device))

        self.layers = []
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                                         nn.BatchNorm2d(128),
                                         nn.ELU()))
        self.layers.append(nn.Sequential(nn.ConvTranspose2d(128, 64, 5, stride=2),
                                         nn.BatchNorm2d(64),
                                         nn.ELU()))
                           
        self.target_output_size = target_output_size
        assert(self.target_output_size in (30, 64))
        self._obs_size = self.target_output_size ** 2

        self.event_shape = (3, self.target_output_size, self.target_output_size)
        self.event_dim = len(self.event_shape)
        
        if self.mixture: raise NotImplementedError('foo')
        else:
            if self.target_output_size == 64:
                self.layers.append(nn.Sequential(nn.ConvTranspose2d(64, 32, 6, stride=2),
                                                 nn.BatchNorm2d(32),
                                                 nn.ELU()))
                self.layers.append(nn.Sequential(nn.ConvTranspose2d(32, 3, 6, stride=2),
                                                 nn.ELU(),
                                                 nn.ConvTranspose2d(3, 3, 1, stride=1)))
            elif self.target_output_size == 30:
                self.layers.append(nn.Sequential(nn.ConvTranspose2d(64, 3, 6, stride=2),
                                                 nn.ELU(),
                                                 nn.ConvTranspose2d(3, 3, 1, stride=1)))
                
            else:
                raise NotImplementedError(self.target_output_size)

        # Treat observations as normalized [0, 1]
        self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, latent_dist_sample, future_actions=None, batch_shape=None, agent_pose=None):
        """p(o_{t:t+K} | s_t, a_{t-1:t+K-1})

        :param latent_dist_sample: (B, D)
        :param future_actions: (B, k_steps, A)
        :param batch_shape: optional batch shape, otherwise defaults to (B,)
        :returns: batch_shape + event_shape distribution
        :rtype: 

        """
        # Add B=1
        if latent_dist_sample.ndim == 1: latent_dist_sample = latent_dist_sample[None]
        # (B, D)
        assert(latent_dist_sample.ndim == 2)
        if self.observe_agent_pose:
            assert(agent_pose is not None)
            assert(agent_pose.ndim == 2)
            input_repr = torch.cat((latent_dist_sample, agent_pose), -1)
        else:
            input_repr = latent_dist_sample
            
        assert(input_repr.isfinite().all().item())

        # Partition off the elements of latent state that we shouldn't use for decoding.
        input_repr_view = input_repr.view(-1, input_repr.shape[-1])[..., :self.state_partition_index]

        # Embed the latent reprs
        hidden = self.fc1(input_repr_view)
        
        # Reshape to enable deconv.
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        # Decode the embeddings to image means.
        observation_mean = self.layers(hidden)

        assert(observation_mean.isfinite().all().item())

        # Undo the pre-deconv batching.        
        if batch_shape is not None:
            # Reshape obs mean to some batch shape if it's provided.
            assert(latent_dist_sample.shape[0] == np.prod(batch_shape))
            observation_mean = observation_mean.view(batch_shape + self.event_shape)
        else:
            observation_mean = observation_mean.view((-1,) + self.event_shape)

        # Let's assume the pixel space is [0, 1]. We don't want out stddev to grow above 1 then.
        if self.learnable_stddev:
            stddev = torch.clamp(torch.exp(self.log_stddev), self.min_std_dev, 1.)
            dist = SliceableIndependentOfNormal(MeanSamplingNormal(loc=observation_mean, scale=stddev, validate_args=True), reinterpreted_batch_ndims=self.event_dim)
        else:
            with Timer('obs model dist inst'):
                dist = SliceableIndependentOfNormal(MeanSamplingNormal(loc=observation_mean, scale=1., validate_args=True), reinterpreted_batch_ndims=self.event_dim)
        self.most_recent_dist = dist
        return dist

class SequentialObservationFeaturizer(ABC):
    @abstractmethod
    def online(self, obs):
        """ 

        :param obs: (C, H, W)
        :returns: 
        :rtype: 

        """

    @abstractmethod
    def batch(self, obs_TBO):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def observation_size(self):
        pass
    
class IndependentSequentialObservationFeaturizer(SequentialObservationFeaturizer):
    @classu.pt_member_initialize
    def __init__(self, observation_featurizer, pfactor=50):
        pass

    @property
    def observation_size(self):
        return self.observation_featurizer.embedding_size

    def reset(self):
        pass

    def online(self, obs):
        return self.observation_featurizer(obs)

    def batch(self, observations_TBO):
        return torch.stack([self.online(x) for x in observations_TBO], 0)

class VisualObservationFeaturizer(torch.nn.Module):
    __constants__ = ['embedding_size']
    
    def __init__(self, embedding_size, activation_function='elu', input_channels=3, input_size=64, conv_flat_size=1024, as_list=True, flatten=False):
        super().__init__()
        self.flatten = flatten

        self.embedding_size = embedding_size
        self.input_size = input_size
        self.layers = []

        
        self.layers.append(nn.Sequential(nn.Conv2d(3, 32, 4, stride=2),
                                         nn.BatchNorm2d(32),
                                         nn.ELU()))

        if self.flatten:
            self.layers.append(nn.Sequential(nn.Conv2d(32, 64, 4, stride=2),
                                             nn.BatchNorm2d(64),
                                             nn.ELU()))

            if self.input_size == 64:
                self.layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, stride=2),
                                                 nn.BatchNorm2d(128),
                                                 nn.ELU()))
                self.layers.append(nn.Sequential(nn.Conv2d(128, 256, 4, stride=2),
                                                 nn.BatchNorm2d(256),
                                                 nn.ELU()))
            elif self.input_size == 30:
                self.layers.append(nn.Sequential(nn.Conv2d(64, 256, 4, stride=2),
                                                 nn.BatchNorm2d(256),
                                                 nn.ELU()))
            else:
                raise NotImplementedError(self.input_size)
            self.conv_flat_size = conv_flat_size
            self.fc = nn.Identity() if embedding_size == conv_flat_size else nn.Linear(conv_flat_size, embedding_size)
            
        self.layers = nn.Sequential(*self.layers)        
        self.embedding_size = embedding_size

    def forward(self, observation):
        # Require normalized observations!!!
        assert(0. <= observation.max().item() <= 1.)
        # Require the observations to match the expected size.
        assert(observation.shape[-2] == observation.shape[-1] == self.input_size)
        assert(observation.ndim == 4)
        
        B = observation.shape[0]
        hidden = self.layers(observation)
        if self.flatten:
            hidden = hidden.reshape((B, self.conv_flat_size))
            encoding = self.fc(hidden)
            return encoding
        else:
            return hidden

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                                  out_channels=self.hidden_dim, # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    # def init_hidden(self, batch_size):
    #     return (torch.autograd.Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next    

class SliceableDeque(collections.deque):
    def __getitem__(self, index):
        try:
            return collections.deque.__getitem__(self, index)
        except TypeError:
            return type(self)(itertools.islice(self, index.start, index.stop, index.step))    

class VBFModelWrapper(Wrapper):
    augmentation_key = 'vbf_posterior'
    posterior_render_key = 'posterior_render'
    prior_render_key = 'prior_render'
    
    def __init__(self, vbf_model, env, render_posterior=False, render_prior=False, augmentation_mode='mean_stddev', multistep_indices=[], bayesian_vbf_model=None, N_posterior_samples=7,
                 compute_surprise=False):
        super().__init__(env)
        assert(augmentation_mode in {'mean_stddev', 'mixture_mean_stddev', 'ekf_mean_stddev', 'mean', 'sample'})

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise NotImplementedError

        self.bayesian_vbf_model = bayesian_vbf_model
        self.N_posterior_samples = N_posterior_samples
        self.multistep_indices = multistep_indices
        self.augmentation_mode = augmentation_mode
        self.vbf_model = vbf_model
        self.render_posterior = render_posterior
        self.render_prior = render_prior
        self.compute_surprise = compute_surprise
        
        dict_kwargs = dict(self.observation_space.spaces.items())
        if self.vbf_model.model_kwargs['observe_agent_pose']:
            agent_pose_size = env.observation_space['agent_pose'].sample().size
            dict_kwargs[self.augmentation_key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(vbf_model.belief_distribution_size + agent_pose_size,))
        else:
            dict_kwargs[self.augmentation_key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(vbf_model.belief_distribution_size,))
        self.observation_space = gym.spaces.Dict(**dict_kwargs)
        if self.vbf_model.model_kwargs['observe_agent_pose']:
            assert(len(multistep_indices) == 0) , "Cant forecast when require agent pose to decode"
    @property
    def wrapper_bfs(self):
        return self._wrapper_bfs

    @property
    def env_timestep(self):
        return self._t

    def reset(self, **kwargs):
        with torch.no_grad():
            B = 1
            self._t = 0
            self.a_tm1 = None
            observation_0_dict = self.env.reset(**kwargs)
            # Set the model to be in evaluation mode.
            self.vbf_model.eval()
            obs_0 = ptu.image_from_numpy(observation_0_dict[self.vbf_model.observation_key][None])

            if self.vbf_model.model_kwargs['observe_agent_pose']:
                agent_pose_0 = ptu.from_numpy(np.asarray(observation_0_dict['agent_pose'])[None])
            else:
                agent_pose_0 = None
            
            self._wrapper_bfs = self.vbf_model.belief_filter_reset(B, K=self.vbf_model.model_kwargs['K_particles_infer'], obs_0=obs_0, track_internally=False, agent_pose_0=agent_pose_0)

            if self.bayesian_vbf_model is not None:
                self.bayesian_vbf_model.resample_ensemble_parameters()
                for em in self.bayesian_vbf_model.ensemble:
                    em.belief_filter_reset(B, K=self.vbf_model.model_kwargs['K_particles_infer'], obs_0=obs_0, track_internally=False)

            self.n_multistep = max(self.multistep_indices) + 1 if len(self.multistep_indices) else 0
            if len(self.multistep_indices):
                assert(min(self.multistep_indices) >= 1)

            # Subset of previous beliefs
            self.multistep_latent_dists = collections.deque(maxlen=self.n_multistep)
            self.action_history = SliceableDeque(maxlen=self.n_multistep)
            for _ in range(self.n_multistep): self.multistep_latent_dists.append(self.wrapper_bfs.posterior_latent_dists[-1])
            aug_obs, pv, o_t_pt = self.step_filter(observation_0_dict, {})
            return aug_obs 

    def step(self, action):
        with torch.no_grad():
            # Ensure 1D for the underlying environment.
            action = np.atleast_1d(action)
            # Ensure 2D for the filter (i.e. B=1)
            self.a_tm1 = ptu.from_numpy(np.atleast_2d(action))
            # Actions are stored [t-1, t-2, ... t-1-maxlen].
            self.action_history.appendleft(self.a_tm1)        
            observation, reward, done, info = self.env.step(action)
            obs, pv, o_t_pt = self.step_filter(observation, info)
            if self.bayesian_vbf_model is not None:
                info[f'vbf_prediction_variance'] = pv
            return obs, reward, done, info

    def _add_info(self, info, a_tm1, o_t_pt):
        if self.compute_surprise:
            with PreserveRolloutState(self.vbf_model.prior_model):
                K = self.vbf_model.model_kwargs['K_particles_infer']

                # z_t^k ~ p(z_t | z_t-1^k, a_t-1), z_t-1^k ~ q(z_t | o_1:t)
                # log p(o_t | ...) \approx log-sum-exp log p(o_t | z_t^k) ) - log K
                posterior_tm1s = self._wrapper_bfs.posterior_latent_dists[-1]
                posterior_tm1_belief = posterior_tm1s.belief_distribution

                z_tm1_samples = posterior_tm1_belief.sample((K,)).transpose(0,1)
                # Package the resampled particles and the previous actions.
                za_tm1_and_agent_pose_t = dict(latent_dist_sample_tm1=z_tm1_samples, action_tm1=self.vbf_model._get_action_repr(a_tm1), agent_pose_t=None)
                prior_t = self.vbf_model.prior_model.forward(**za_tm1_and_agent_pose_t)
                prior_t_samples = prior_t.state_sample
                prior_t_samples_batch = prior_t_samples.view((K, -1))
                p_o_t = self._wrapper_bfs.observation_model(prior_t_samples_batch)
                log_p_o_t = p_o_t.log_prob(o_t_pt)
                # NB the negative.
                surprise_t = -1 * ptu.to_numpy(torch.logsumexp(log_p_o_t, -1)).item()
                info['vbf_surprise_t'] = surprise_t
    
    def sample_obs_from_latent_dists_belief_dist(self, latent_dist, agent_pose=None, N=1, **kwargs):
        assert(N == 1)
        # NB it's the belief distribution
        return ptu.to_numpy(torch.clamp(self.wrapper_bfs.observation_model(latent_dist.belief_distribution.sample().flatten()[None], agent_pose=agent_pose).sample(), 0., 1.))

    def sample_obs_from_latent_dist(self, latent_dist, agent_pose=None, N=1, **kwargs):
        assert(N == 1)
        return ptu.to_numpy(torch.clamp(self.wrapper_bfs.observation_model(latent_dist.distribution.sample().flatten()[None], agent_pose=agent_pose).sample(), 0., 1.))

    def likeliest_obs_from_latent_dist(self, latent_dist, agent_pose=None, N=1, **kwargs):
        assert(N == 1)
        return ptu.to_numpy(torch.clamp(self.wrapper_bfs.observation_model(latent_dist.distribution.mean.flatten()[None], agent_pose=agent_pose).mean, 0., 1.))

    def step_filter(self, o_t_dict, info):
        assert(isinstance(o_t_dict, dict))
        assert(self.augmentation_key not in o_t_dict)
        assert(not self.vbf_model.training)
        o_t_dict_augmented = copy.copy(o_t_dict)
        o_t = o_t_dict[self.vbf_model.observation_key]

        # When t=0, we don't have an action yet, although we will call step_filter() in order to transform the first observation.
        have_previous_action = self.a_tm1 is not None
        assert(self.wrapper_bfs.posterior_latent_dists[-1].B == 1)
        assert(self.wrapper_bfs.prior_latent_dists[-1].B == 1)

        prediction_variance = 0.0

        def replace_with_first_image(od, k): od[k] = od[k][0]

        if self.vbf_model.model_kwargs['observe_agent_pose']:
            agent_pose_t = np.asarray(o_t_dict['agent_pose'])[None]
            agent_pose_t_tensor = ptu.from_numpy(agent_pose_t)
        else:
            agent_pose_t_tensor = agent_pose_t = None        

        o_t_pt = ptu.image_from_numpy(o_t)
        if have_previous_action:
            # Add some info before stepping the filter.
            self._add_info(info, self.a_tm1, o_t_pt)              

            assert(self._t >= 1)
            bf_kwargs = dict(o_t=o_t_pt, a_tm1=self.a_tm1, t=self._t, bfs_t=self._wrapper_bfs, agent_pose_t=agent_pose_t_tensor)

            # ----------------------------------------
            # The primary step of the belief filter
            # ----------------------------------------
            self.vbf_model.belief_filter_single_step(**bf_kwargs)      

            # Prior-specific epistemic uncertainty:
            if self.bayesian_vbf_model is not None:
                with Timer("epistemic uncertainty time", enabled=False):
                    predictions = []
                    # NB we're using the old and new particles from the main BFS to define the inputs to the forward and the targets of the dists.
                    z_and_a_tm1 = dict(latent_dist_sample_tm1=self._wrapper_bfs.particles[-2], action_tm1=self.vbf_model._get_action_repr(self.a_tm1))
                    latent_state_samples_t = self._wrapper_bfs.particles[-1]
                    for em in self.bayesian_vbf_model.ensemble:
                        prior_latent_dist_t = em.prior_model.forward(**z_and_a_tm1)
                        predictions.append(prior_latent_dist_t.distribution_detached.log_prob(latent_state_samples_t))
                    prediction_variance = np.var(ptu.to_numpy(torch.cat(predictions)))
                    # log.debug(f"Prediction Variance: {prediction_variance:.4f}")
                    
            # Beliefs are stored [t, t-1, ... t-maxlen]
            self.multistep_latent_dists.appendleft(self.wrapper_bfs.posterior_latent_dists[-1])
            for index_index, multistep_index in enumerate(self.multistep_indices):
                # retrieve t-multistep_index (multistep_index >= 1)
                multistep_latent_dist = self.multistep_latent_dists[multistep_index]
                # Actions are stored [t-1, t-2, ... t-1-maxlen]. Retrieve the slice (t-1:t-1-multistep_index)
                action_traj = list(self.action_history[:multistep_index])[::-1]
                future_latent_dist = self.vbf_model.forecast_latent_dist_with_action_plan(multistep_latent_dist.distribution.sample(), action_traj)[-1]
                key = f'image_multistep_prior_{index_index}'
                o_t_dict_augmented[key] = self.sample_obs_from_latent_dist(future_latent_dist, agent_pose=None, N=1)[0]
                if self.vbf_model.observation_model.k_steps > 1:
                    replace_with_first_image(o_t_dict_augmented, key)
                    
        else:
            assert(self._t == 0)

        if self.render_prior:
            N = 1
            assert(N==1)
            image_prior_samples = self.likeliest_obs_from_latent_dist(self.wrapper_bfs.prior_latent_dists[-1], agent_pose=self.wrapper_bfs.agent_poses[-1], N=N)
            o_t_dict_augmented['image_prior_render'] = image_prior_samples[0]

            if self.vbf_model.observation_model.k_steps > 1:
                replace_with_first_image(o_t_dict_augmented, 'image_prior_render')
            assert(o_t_dict_augmented['image_prior_render'].ndim == 3)

        # Prior for the next timestep is the posterior for t (note that the 0 index is just a dimensionality reduction, not list indexing)
        if self.augmentation_mode == 'mixture_mean_stddev':
            o_t_policy = ptu.to_numpy(self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution_parameters.squeeze())
        elif self.augmentation_mode == 'mean_stddev':
            o_t_policy = ptu.to_numpy(self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution_parameters.squeeze())            
        elif self.augmentation_mode == 'ekf_mean_stddev':
            o_t_policy = ptu.to_numpy(self.vbf_model.ekf_belief.distribution_parameters)            
        else:
            raise ValueError(self.augmentation_mode)
            
        # Always 1-D
        o_t_dict_augmented[self.augmentation_key] = o_t_policy.ravel()
        
        if self.render_posterior:
            o_t_dict_augmented['image_posterior_render'] = self.sample_obs_from_latent_dists_belief_dist(self.wrapper_bfs.posterior_latent_dists[-1],
                                                                                                         agent_pose=self.wrapper_bfs.agent_poses[-1])[0]
            if self.vbf_model.observation_model.k_steps > 1:
                replace_with_first_image(o_t_dict_augmented, 'image_posterior_render')
            assert(o_t_dict_augmented['image_posterior_render'].ndim == 3)
        self._t += 1
        return o_t_dict_augmented, prediction_variance, o_t_pt

if __name__ == '__main__':
    raise RuntimeError("Not executable")
