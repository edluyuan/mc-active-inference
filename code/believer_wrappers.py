
import collections
import numpy as np
import gym
from gym import ObservationWrapper, RewardWrapper, Wrapper
from gym.wrappers import TransformObservation
import gym.spaces
import itertools
import logging
import numpy as np
import os
import pdb
import scipy.stats
import skimage
import pdb
import torch
from torch import nn
from torch.nn import functional as F
import torch_optimizer as optim
from torch.distributions import Normal, Independent, MultivariateNormal, OneHotCategorical, Categorical, MixtureSameFamily, RelaxedOneHotCategorical, ContinuousBernoulli, Dirichlet, LowRankMultivariateNormal
import welford
import tqdm

import railrl.misc.class_util as classu
import railrl.torch.pytorch_util as ptu
from railrl.torch.core import np_to_pytorch_batch
import railrl.torch.ic2.marginal.marginals as marginals
import railrl.torch.ic2.planet.vbf as vbf

log = logging.getLogger(os.path.basename(__file__))
meannpitem = lambda x: ptu.to_numpy(x.mean()).item()

class DictifyObservation(TransformObservation):
    def __init__(self, env, name):
        def dictify(obs): return {name: obs}
        super().__init__(env, dictify)
        self.observation_space = gym.spaces.Dict({name: self.observation_space})

class CombineObservations(TransformObservation):
    def __init__(self, env, names, jointname=None):
        if jointname is None:
            jointname = '_'.join(names)
        
        def combine(obs):
            obss = [obs[name] for name in names]
            joint = np.concatenate(obss, axis=-1)
            obs[jointname] = joint
            return obs

        super().__init__(env, combine)
        xd = dict(self.observation_space.spaces.items())
        low = np.concatenate([xd[name].low for name in names], axis=-1)
        high = np.concatenate([xd[name].high for name in names], axis=-1)
        # shape = (sum(sum([xd[name].shape for name in names], ())),)
        xd[jointname] = gym.spaces.Box(low=low, high=high)
        self.observation_space = gym.spaces.Dict(**xd)

class DiscretizeEnv(Wrapper):
    def __init__(self, env, num_bins):
        super().__init__(env)
        self.env = env

        # TODO assumes env action space is actually continuous.
        low = self.env.action_space.low
        high = self.env.action_space.high
        action_ranges = [np.linspace(low[i], high[i], num_bins) for i in range(len(low))]
        
        self.idx_to_continuous_action = [np.array(x) for x in itertools.product(*action_ranges)]
        self.action_space = gym.spaces.Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        if isinstance(action, (torch.Tensor, np.ndarray)):
            action = action.item()
        assert(self.action_space.contains(action))
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)


class EyeClosingWrapper(Wrapper):
    def __init__(self, env, oname):
        super().__init__(env)
        
        action_space_is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        assert(action_space_is_discrete)

        xd = dict(self.observation_space.spaces.items())
        assert(oname in xd)
        
        self.oname = oname
        self.close_eye_action = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.close_eye_action + 1)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
            
        assert(isinstance(action, int))        
        eyes_closed = action == self.close_eye_action
        
        if eyes_closed:
            # Default NOOP to 5... TODO
            action = 5
        
        dict_obs, reward, done, info = super().step(action)
        assert(self.oname in dict_obs)
        if eyes_closed:
            dict_obs[self.oname] = np.zeros_like(dict_obs[self.oname])
        info['eyes_closed'] = eyes_closed
        return dict_obs, reward, done, info
        
class ResizeImageObservationInDict(TransformObservation):
    def __init__(self, env, oname, newshape, old_key=None):
        full_newshape = (3,) + newshape
        def resize(obs_d):
            # (A, B, 3)
            orig = obs_d[oname]
            orig_dtype = orig.dtype
            if old_key is not None:
                obs_d[old_key] = orig

            if orig.shape == full_newshape:
                return obs_d
            elif orig.shape == newshape + (3,):
                obs_d[oname] = orig.transpose((2, 0, 1))
            elif orig.shape[-1] == 3 or orig.shape[0] == 3:
                if orig.shape[0] == 3:
                    orig = orig.transpose((1, 2, 0))
                # e.g. if newshape=(64, 64, 3) -> (3, 64, 64)
                assert(orig.shape[:2] != newshape)
                need_to_antialias = newshape[0] < orig.shape[0]
                obs_R = skimage.transform.resize(orig, newshape, anti_aliasing=need_to_antialias)
                if orig_dtype == np.uint8:
                    obs_R = skimage.img_as_ubyte(obs_R)
                obs_R = obs_R.transpose((2, 0, 1))
                # Overrwrite original.
                obs_d[oname] = obs_R
                assert(obs_R.shape == full_newshape)
                ###### end ########
                
                return obs_d
            else:
                raise NotImplementedError("image dim")
        super().__init__(env, resize)

        orig_space = env.observation_space[oname]
        if old_key is not None:
            self.observation_space.spaces[old_key] = self.observation_space.spaces[oname]
        self.observation_space.spaces[oname] = gym.spaces.Box(orig_space.low.min(), orig_space.high.max(), shape=full_newshape, dtype=orig_space.dtype)

    def reset(self):
        return super().reset()

class EnvInfoRewardWrapper(gym.Wrapper):
    def __init__(self, key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._info_key = key
        
    def step(self, action):
        o, r, d, i = super().step(action)
        assert(self._info_key in i)
        r_new = i[self._info_key]
        assert('original_reward' not in i)
        i['original_reward'] = r
        r = r_new
        return o,  r_new, d, i

class RNDWrapper(gym.Wrapper):
    @classu.member_initialize
    def __init__(self, env, rnd_model, obs_key):
        """
        :param env: 

        """
        super().__init__(env)
        self.rnd_model = rnd_model
        self.obs_key = obs_key

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            rnd_bonus = meannpitem(self.rnd_model(ptu.from_numpy(dict_obs[self.obs_key])))
            reward = rnd_bonus
            return dict_obs, reward, done, info

class SurpriseRewardWrapper(Wrapper):
    
    @classu.member_initialize
    def __init__(self, env, marginal_obs_key, policy_obs_key, marginal, include_suff_stats=True):
        """
         Wrapper to change the reward of env to the log pdf of `marginal` fit online to the observations of 
        `marginal_obs_key` each episode.

        :param env: 
        :param marginal_obs_key: 
        :param marginal: 
        :returns: 
        :rtype: 

        """
        super().__init__(env)
        assert(hasattr(marginal, 'logpdf'))
        assert(hasattr(marginal, 'fit'))
        assert(hasattr(marginal, 'sufficient_statistics'))
        
        aug_idx = policy_obs_key.rfind('_augmented')
        xd = dict(self.observation_space.spaces.items())
        
        # Force policy_obs_key to contain 'augmented', i.e. force the SMIRL policy to observe augmented things
        # assert(aug_idx > 0)
        if aug_idx > 0:
            self._pre_augmented_policy_obs_key = policy_obs_key[:aug_idx]
            # Ensure we'll be creating a new observation subspace for the policy.
            assert(policy_obs_key not in xd)        
        else:
            assert(not self.include_suff_stats)
            assert(policy_obs_key in xd)        
        
        if self.include_suff_stats:
            orig_space = xd[self._pre_augmented_policy_obs_key]
            suff_size = self.marginal.sufficient_statistics().size
            if isinstance(orig_space, gym.spaces.multi_discrete.MultiDiscrete):
                new_high = np.concatenate((orig_space.nvec, np.full((suff_size,), np.inf)),axis=-1)
                new_space = type(orig_space)(new_high)
            else:
                new_low = np.concatenate((orig_space.low.ravel(), np.full((suff_size,), -np.inf)),axis=-1)
                new_high = np.concatenate((orig_space.high.ravel(), np.full((suff_size,), np.inf)),axis=-1)
                new_space = type(orig_space)(low=new_low, high=new_high)

            # Ensure the wrapped environment allocates a new space, i.e. doesn't overwrite the inner environment's obs space.
            xd[self.policy_obs_key] = new_space
            self.observation_space = gym.spaces.Dict(**xd)

    def observation(self, dict_obs):
        if self.include_suff_stats:
            # Create an augmented observation from the pre-augmented observation
            augmented_obs = np.concatenate((dict_obs[self._pre_augmented_policy_obs_key].ravel(),
                                            self.marginal.sufficient_statistics()), axis=-1)
            # Store the augmented observation in dict observations.
            dict_obs[self.policy_obs_key] = augmented_obs
        assert(self.policy_obs_key in dict_obs)
        return dict_obs
    
    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the current observation.

        :param action: 
        :returns: 
        :rtype: 

        """
        
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)

        # Evaluate the agent's surprise.
        marginal_obs = dict_obs[self.marginal_obs_key].flatten()
        log_p = self.marginal.logpdf(marginal_obs[None]).item()

        # assert('original_reward' not in info)
        info['original_reward'] = reward

        # Dimensionality normalization makes reward scale smaller -- easier to set reward-scaling sensitive params.
        reward = log_p / float(marginal_obs.size)
        info['marginal_entropy'] = log_p

        # Fit the marginal after computing the agent's surprise.
        self.marginal.fit(marginal_obs[None], online=True)
        
        return self.observation(dict_obs), reward, done, info

    def reset(self):
        # Reset the marginal.
        self.marginal.reset()
        
        # Reset the environment.
        return self.observation(super().reset())


class KLPosteriorPriorEpisodeGainRewardWrapper(Wrapper):
    # TODO JW: is this right?

    @classu.member_initialize
    def __init__(self, env):
        """

        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """

        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)

        # This is provided by the planet model wrapper

        kl = info['kl(posterior,init)']
        assert('original_reward' not in info)
        info['original_reward'] = reward

        # Gather information by surprising the prior.
        reward = kl

        return dict_obs, reward, done, info

class HMMObjectiveRewardWrapper(Wrapper):
    
    @classu.member_initialize
    def __init__(self, env, info_key):
        """

        :param env: 

        """
        super().__init__(env)
        self.info_key
    
    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)
        hmm_value = info[self.info_key]
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward = hmm_value
        return dict_obs, reward, done, info    
    

class CrossObservationPredictionWrapper(Wrapper):
    class Trainer:
        @classu.member_initialize
        def __init__(self, env_wrapped, optimizer_klass=optim.RAdam, learning_rate=1e-3, grad_clip_norm=1000):
            self.params_list = list(self.env_wrapped.mlp.parameters())
            self.optimizer = optimizer_klass(self.params_list, lr=self.learning_rate)

        def fit_minibatch(self, batch):
            self.optimizer.zero_grad()
            losses = self.env_wrapped.compute_crossobs_losses(batch=batch)
            losses['crossobs_loss'].backward()
            nn.utils.clip_grad_norm_(self.params_list, self.grad_clip_norm, norm_type=2)
            self.optimizer.step()
            # with torch.no_grad():
            #     losses_poststep = self.env_wrapped.compute_crossobs_losses(batch=batch)
            # losses['crossobs_mse_post'] = losses_poststep['mse']
            return ptu.to_numpy(losses)

    class ReplayBufferTrainer:
        @classu.member_initialize
        def __init__(self, replay_buffer, crossobs_pred_trainer, B):
            pass

        def fit_minibatch(self):
            train_data = self.replay_buffer.random_observation_batch(batch_size=self.B)
            
            # Convert numpy arrays to torch.Tensors on the default device.
            train_data = np_to_pytorch_batch(train_data)

            losses = self.crossobs_pred_trainer.fit_minibatch(train_data)
            return losses
    
    @classu.member_initialize
    def __init__(self, env, x_key, y_key, mlp=None, hiddens=None, classify=True, all_position_inds=[[0, 1], [3, 4]]):
        """An environment that maintains an MLP to perform cross-observation prediction y_hat=f(x)

        :param env: 
        :param x_key: 
        :param y_key: 
        :param mlp: 
        :param hiddens: 
        :param classify: 
        :param all_position_inds: TODO hardcodsed
        :returns: 
        :rtype: 

        """
        super().__init__(env)

        obs = self.reset()
        x = obs[x_key]
        y = obs[y_key]
        self.crossobs_keys = [x_key, y_key]

        if self.classify:
            # These are the sizes of each dimension of observation space
            one_hot_sizes = self.observation_space[y_key].nvec
            # Include the 0th index in the sizes
            one_hot_indices = np.concatenate(([0], np.cumsum(one_hot_sizes)), axis=-1)
            # Build slices to later use for a flattened sequence of one-hot vectors
            self.one_hot_slices = [slice(one_hot_indices[i], one_hot_indices[i+1]) for i in range(len(one_hot_indices) - 1)]
            # This will be the size of the full prediction.
            output_size = np.sum(one_hot_sizes)
        else:
            output_size = y.size
        self._x_size = x.size
        
        if self.mlp is None:
            from railrl.torch.networks import Mlp, SymmetricEncodingMlp
            # Default to a single hidden layer of 200 units.
            if hiddens is None: hiddens = [200]

            if x.ndim > 1:
                # Treat multidimensional objects as something we want to compute a symmetric encoding of (permutation invariant)
                mlp_kwargs = dict(hidden_sizes=hiddens, output_size=output_size, hidden_activation=torch.tanh)
                encoding_size = 200
                self.mlp = SymmetricEncodingMlp(input_size=x.shape[-1], output_size=encoding_size, mlp_kwargs=mlp_kwargs, axis=-2)
                self.mlp.to(ptu.device)
            else:
                self.mlp = Mlp(hiddens, output_size=output_size, input_size=x.size, hidden_activation=torch.tanh)
                self.mlp.to(ptu.device)


    def predict(self, x):
        if self.classify:
            scores = self.mlp(x)
            # Extract the scores (unnormalized!) for each of the objects to predict.
            # i.e. these are the N_i per-class scores of each q_i(- | filtering_state).
            per_target_log_predictions = [scores[..., _] for _ in self.one_hot_slices]
            return per_target_log_predictions
        else:
            return self.mlp(x)

    def _build_2d_belief_maps(self, y_hat_probs):
        belief_maps = []
        for pi_inds in self.all_position_inds:
            belief_maps.append(self._build_2d_belief_map(pi_inds, y_hat_probs))
        return belief_maps
                
    def _build_2d_belief_map(self, position_inds, y_hat_probs):
        assert(len(position_inds) == 2)
        x = y_hat_probs[position_inds[0]]
        y = y_hat_probs[position_inds[1]]
        belief_map = np.outer(y, x)
        # TODO this assumes that there's a one-pixel outer edge / wall with 0-probability values of belief
        # belief_map = skimage.util.pad(belief_map, [(1,1), (1,1)])
        belief_map = np.pad(belief_map, 1, constant_values=0.)
        return belief_map
    
    def step(self, action):
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)

        x = dict_obs[self.x_key]
        assert(x.size == self._x_size)

        with torch.no_grad():
            y_hat = self.predict(ptu.from_numpy(x))
            
        y = dict_obs[self.y_key]
        if self.classify:
            # NB that we assume the environment tells us what the 'discrete_state_labels' are!
            state_labels = self.discrete_state_labels
            # List of q_i(- | filtering_state) for all i.
            y_hat_probs = [ptu.to_numpy(torch.softmax(_, axis=-1)) for _ in y_hat]
            belief_maps = self._build_2d_belief_maps(y_hat_probs)
            for _, belief_map in enumerate(belief_maps):
                info[f'belief_map_{_}'] = belief_map
            assert(len(state_labels) == len(y_hat_probs) == len(y))
            for y_i, y_hat_i, label_i in zip(y, y_hat_probs, state_labels):
                # Measure the entropy of q_i(- | filtering_state)
                info[f'entropy_of_q_{label_i}_dist'] = scipy.stats.entropy(y_hat_i)
                # Measure q_i(s_i | filtering_state).
                info[f'model_probability_of_{label_i}'] = y_hat_i[y_i]
        else:
            state_labels = self.continuous_state_labels
            y_hat = ptu.to_numpy(y_hat)
            se = np.power(y - y_hat, 2)
            mse = se.mean()
            info[f'MSE(f({self.x_key}; theta), {self.y_key})'] = mse
            assert(len(state_labels) == len(se))
            for i, (sl_i, se_i) in enumerate(zip(state_labels, se)):
                info[f'SE(f({self.x_key}; theta)_{i}, {sl_i})'] = se_i
        return dict_obs, reward, done, info

    def compute_crossobs_losses(self, batch):
        x = batch[self.x_key]
        # NB hack because current ic2 wrapper overrwites the observation to be larger.
        x = x[..., :self.mlp.input_size].squeeze()
        
        y_hat = self.predict(x)
        y = batch[self.y_key]
        if self.classify:
            # Integer targets denote classes.
            y = y.long()
            # Split out the targets to have one per prediction problem.
            y_targets = torch.split(y, 1, dim=-1)
            assert(len(y_targets) == len(y_hat))
            # We will minimize a sum of cross entropies.
            loss_func = torch.nn.CrossEntropyLoss().to(ptu.device)
            loss_list = []
            for y_target, y_hat_i in zip(y_targets, y_hat):
                # NB the cross entropies use unnormalized scores.
                loss_list.append(loss_func(target=y_target.reshape(-1), input=y_hat_i))
            cross_entropy_sum = sum(loss_list)
            losses = dict(crossobs_loss=cross_entropy_sum)
        else:
            se = torch.pow(y - y_hat, 2)
            mse = se.mean()
            losses = dict(crossobs_loss=mse)
        return losses

    @classmethod
    def multi_init(wrapper_cls, env, replay_buffer, x_key, y_key, B, mlp=None):
        env_wrapped = wrapper_cls(env=env, x_key=x_key, y_key=y_key, mlp=None)
        trainer = wrapper_cls.Trainer(env_wrapped=env_wrapped)
        rb_trainer = wrapper_cls.ReplayBufferTrainer(replay_buffer=replay_buffer, crossobs_pred_trainer=trainer, B=B)
        return dict(env=env_wrapped, trainer=trainer, rb_trainer=rb_trainer)

class CrossObservationImagePredictionWrapper(Wrapper):
    class Trainer:
        @classu.member_initialize
        def __init__(self, env_wrapped, optimizer_klass=optim.RAdam, learning_rate=1e-3, grad_clip_norm=1000):
            self.params_list = list(self.env_wrapped.observation_model.parameters())
            self.optimizer = optimizer_klass(self.params_list, lr=self.learning_rate)

        def fit_minibatch(self, batch):
            """

            :param batch: pytorchified batch
            :returns: 
            :rtype: 

            """
            # pdb.set_trace()
            self.optimizer.zero_grad()
            losses = self.env_wrapped.compute_crossobs_losses(batch=batch)
            losses['crossobs_loss'].backward()
            nn.utils.clip_grad_norm_(self.params_list, self.grad_clip_norm, norm_type=2)
            self.optimizer.step()
            return ptu.to_numpy(losses)        

    class ReplayBufferTrainer:
        @classu.member_initialize
        def __init__(self, replay_buffer, crossobs_pred_trainer, B):
            pass

        def fit_minibatch(self):
            train_data = self.replay_buffer.random_observation_batch(batch_size=self.B)
            
            # Convert numpy arrays to torch.Tensors on the default device.
            train_data = np_to_pytorch_batch(train_data)

            # Feed the pytorchified minibatch to the trainer.
            losses = self.crossobs_pred_trainer.fit_minibatch(train_data)
            return losses

        def fit_epochs(self, writer, number, visualize=False, mode='pretrain'):
            if number > 1:
                for i in tqdm.trange(int(round(number))): res = self.fit_epoch(writer)
                return res
            elif number > 0:
                return self.fit_partial_epoch(writer, number)
            else:
                raise ValueError(number)        

        def fit_epoch(self, writer):
            return self.fit_partial_epoch(writer, fraction=1.)

        def fit_partial_epoch(self, writer, fraction=0.5):
            assert(0 < fraction <= 1.)
            n_approx_windows = self.replay_buffer.get_approximate_minibatch_count(batch_size=self.B)
            n_fraction_windows = max(int(round(n_approx_windows * fraction)), 1)
            for i in tqdm.trange(n_fraction_windows):
                ret = self.fit_minibatch()
                for fl in ret.items(): writer.add_scalar(f'Probe_opt/{fl[0]}', fl[1], writer.step)
                writer.increment()
            return ret

        def evaluate_epoch(self, replay_buffer=None):
            if replay_buffer is None: replay_buffer = self.replay_buffer
            stats = collections.defaultdict(list)
            log.info(f"Evaluating probe on epoch of buffer: {replay_buffer}")
            with torch.no_grad():
                for minibatch in replay_buffer.random_observation_batch_without_replacement(batch_size=self.B):
                    minibatch_pt = np_to_pytorch_batch(minibatch)
                    losses = self.crossobs_pred_trainer.env_wrapped.compute_crossobs_losses(batch=minibatch_pt)
                    for k, v in losses.items(): stats[k].append(ptu.to_numpy(v))

            log.debug(f"{len(stats[k])} minibatches evaluated")
            summary_stats = {}
            for k, values in stats.items():
                summary_stats[k + '_mean'] = np.mean(values)
                summary_stats[k + '_sem'] = scipy.stats.sem(values)
            return summary_stats

        def __repr__(self):
            return f"{self.__class__.__name__}(replay_buffer={self.replay_buffer})"
    
    @classu.member_initialize
    def __init__(self, env, x_key, y_key, observation_model):
        """An environment that maintains a observation_model (decoder) to perform cross-observation prediction y_hat=f(x)

        :param env: 
        :param x_key: 
        :param y_key: 
        :param mlp: 
        :param hiddens: 
        :param classify: 
        :param all_position_inds: TODO hardcodsed
        :returns: 
        :rtype: 

        """
        super().__init__(env)
        self.crossobs_keys = [x_key, y_key]
        self.crossobs_observation_model = observation_model

    def predict(self, x):
        return self.crossobs_observation_model(x)

    def step(self, action):
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)

        x = dict_obs[self.x_key]
        y = dict_obs[self.y_key]
        with torch.no_grad():
            y_hat = self.predict(ptu.from_numpy(x))
            # Store the image prediction in the observations
            dict_obs['omniscient_image_prediction'] = ptu.to_numpy(torch.clamp(y_hat.mean[0], 0., 1.))
        return dict_obs, reward, done, info

    def compute_crossobs_losses(self, batch):
        # The batch should already be pytorchified
        x = batch[self.x_key]
        y = batch[self.y_key]
        dist = self.predict(x)
        losses = dict(crossobs_loss=-1*dist.log_prob(y).mean())
        losses['mse'] = torch.nn.functional.mse_loss(dist.mean, y)
        return losses

    @classmethod
    def multi_init(wrapper_cls, env, replay_buffer, x_key, y_key, B, mlp=None):
        env_wrapped = wrapper_cls(env=env, x_key=x_key, y_key=y_key, mlp=None)
        trainer = wrapper_cls.Trainer(env_wrapped=env_wrapped)
        rb_trainer = wrapper_cls.ReplayBufferTrainer(replay_buffer=replay_buffer, crossobs_pred_trainer=trainer, B=B)
        return dict(env=env_wrapped, trainer=trainer, rb_trainer=rb_trainer)
    

class StepCount(int):
    def __init__(self):
        self._count = 0

    @property
    def value(self):
        return self._count

    def increment(self):
        self._count += 1
        
class TotalStepCounter(Wrapper):
    def __init__(self, env, step_count=None):
        super().__init__(env)
        self.step_count = step_count
        if self.step_count is None: self.step_count = StepCount()
    
    def step(self, action):
        self.step_count.increment()
        # log.debug("stepping total step {self.step_count}")
        return super().step(action)

class FlatObsWrapper(Wrapper):
    @classu.member_initialize
    def __init__(self, env, obs_key):
        super().__init__(env)
        self.env = env
        xd = dict(env.observation_space.spaces.items())
        assert(not hasattr(self, 'flat_key'))
        self.flat_key = obs_key + '_flat'
        assert(self.flat_key not in xd)
        orig_space = xd[obs_key]
        print("WARNING: FlatObsWrapper is normalizing flattened observations to have values in [0, 1] for a sketchy fix for image_flat")
        self.orig_space_low = orig_space.low.ravel()
        self.orig_space_high = orig_space.high.ravel()
        xd[self.flat_key] = gym.spaces.Box(low=np.full(self.orig_space_low.shape, 0), high=np.full(self.orig_space_high.shape, 1), dtype=np.float32)
        # xd[self.flat_key] = gym.spaces.Box(low=orig_space.low.ravel(), high=orig_space.high.ravel(), dtype=orig_space.dtype)
        self.observation_space = gym.spaces.Dict(**xd)

    def _augment_with_flattened(self, obs):
        assert(self.flat_key not in obs)
        # obs[self.flat_key] = obs[self.obs_key].ravel()
        obs[self.flat_key] = (obs[self.obs_key].ravel() + self.orig_space_low) / (self.orig_space_high - self.orig_space_low)
        return obs

    def reset(self):
        return self._augment_with_flattened(super().reset())

    def step(self, action):
        ret = super().step(action)
        # obs = ret[0]
        self._augment_with_flattened(ret[0])
        return ret

class VBFAverageBeliefWrapper(Wrapper):
    marginal_obs_key = 'vbf_posterior'
    augmented_marginal_obs_key = 'vbf_posterior_with_suff_stats'

    @classu.member_initialize
    def __init__(self, env, include_suff_stats=True, include_timestep=True, **kwargs):
        """
        :param env: 

        """
        super().__init__(env)
        warn_present(kwargs)

        assert hasattr(self, 'vbf_model')

        self.use_gaussian = self.vbf_model.model_kwargs['posterior_model']['type'].find('normal') >= 0
        self.use_categorical = self.vbf_model.model_kwargs['posterior_model']['type'].find('categorical') >= 0
        if self.use_gaussian:
            pass
        elif self.use_categorical:
            self.mixture_categorical = True
        else:
            raise NotImplementedError("Unknown belief type")

        xd = dict(self.observation_space.spaces.items())
        if self.include_suff_stats:
            orig_space = xd[self.marginal_obs_key]
            suff_size = self.vbf_model.posterior_model.latent_state_size
            if self.use_gaussian: suff_size *= 2
            if self.include_timestep:
                suff_size += 1

            new_low = np.concatenate((orig_space.low, np.full((suff_size,), -np.inf)),axis=-1)
            new_high = np.concatenate((orig_space.high, np.full((suff_size,), np.inf)),axis=-1)
            new_space = type(orig_space)(low=new_low, high=new_high)

            # Ensure the wrapped environment allocates a new space, i.e. doesn't overwrite the inner environment's obs space.
            xd[self.augmented_marginal_obs_key] = new_space
            self.observation_space = gym.spaces.Dict(**xd)
        

    def _average_reset(self):
        if self.use_gaussian:
            self._gaussian_reset()
        elif self.use_categorical:
            self._categorical_reset()
        else:
            raise NotImplementedError
            
    def reset(self):
        ret = super().reset()
        self._average_reset()
        return self._augment_observation(ret)

    def _gaussian_reset(self):
        latent_dist_0 = self.wrapper_bfs.posterior_latent_dists[-1]
        assert(len(self.wrapper_bfs.posterior_latent_dists) == 1)
        assert(latent_dist_0.B == 1)
        self._avg_posterior_dist = latent_dist_0.distribution_detached
        self._marginal_mean_accumulation = self._avg_posterior_dist.mean
        self._mean_sqs = [torch.pow(self._avg_posterior_dist.mean, 2)]
        # NB assumes variance is a batch of vectors (not matrix).
        cov_diag = self._avg_posterior_dist.variance

        # Cover the situation when the distribution wasn't averaged over
        if cov_diag.ndim == 3 and cov_diag.shape[0] == 1 and cov_diag.shape[1] == 1:
            cov_diag = cov_diag.squeeze()[None, :]
        elif cov_diag.ndim == 3 and cov_diag.shape[1] > 1:
            raise NotImplementedError("implement for multisample belief dists")
        else:
            pass
        assert(cov_diag.ndim == 2)
        assert(cov_diag.shape[0] == 1)
        self._cov_diags = [cov_diag]
        self._set_avg_belief_from_gaussian_moment_match()

    def _categorical_reset(self):
        belief_0 = self.current_belief
        if self.wrapper_bfs.posterior_latent_dists[-1].is_belief:
            self._avg_probs = belief_0.base_dist.probs
        else:
            self._avg_probs = belief_0.mean[None]
        self._avg_belief_dist = vbf.IndependentCategoricalLatentStateDistribution(self._avg_probs, is_belief=False).distribution_detached
        if self.mixture_categorical:
            self._probs_sequence = [self.current_belief.component_distribution.base_dist.probs]

    def _augment_observation(self, obs):
        if self.include_suff_stats:
            if self.include_timestep:
                extra = (np.asarray(self.env_timestep)[None],)
            else:
                extra = ()

            if self.use_categorical:
                belief_and_average = np.concatenate((obs[self.marginal_obs_key], ptu.to_numpy(self._avg_probs).ravel()) + extra, -1)
                obs[self.augmented_marginal_obs_key] = belief_and_average
            elif self.use_gaussian:
                belief_and_average = np.concatenate((obs[self.marginal_obs_key], ptu.to_numpy(self._avg_suff_stats).ravel()) + extra, -1)
                obs[self.augmented_marginal_obs_key] = belief_and_average
            else:
                raise NotImplementedError

        return obs

    def _categorical_step(self, ret):
        N = len(self.wrapper_bfs.posterior_latent_dists) - 1
        belief_Np1 = self.current_belief        

        if self.mixture_categorical:
            new_probs = belief_Np1.component_distribution.base_dist.probs
            with torch.no_grad():
                self._probs_sequence.append(new_probs)
                all_probs = torch.cat(self._probs_sequence, 1)
                assert(all_probs.ndim == 4)
                N = all_probs.shape[1]
                components = vbf.IndependentCategoricalLatentStateDistribution(all_probs, is_belief=True).distribution_detached
                categorical_weights = 1/N * torch.ones((1,N), device=ptu.device)
                categorical = Categorical(categorical_weights, validate_args=True)
                dist = MixtureSameFamily(mixture_distribution=categorical, component_distribution=components, validate_args=True)
                self._avg_belief_dist = dist

        else:
            if self.wrapper_bfs.posterior_latent_dists[-1].is_belief:
                new_probs = belief_Np1.base_dist.probs
            else:
                new_probs = belief_Np1.mean[None]
                
            # The averaging version
            self._avg_probs = (self._avg_probs * N + new_probs) / (N + 1)        
            self._avg_belief_dist = vbf.IndependentCategoricalLatentStateDistribution(self._avg_probs, is_belief=True).distribution_detached
        return ret

    def step(self, *args, **kwargs):
        self._previous_avg_belief_dist = self._avg_belief_dist
        
        # Call step of wrapped env.
        obs, reward, done, info = ret = super().step(*args, **kwargs)
       
        if self.use_gaussian:
            self._gaussian_step(ret)
        elif self.use_categorical:
            self._categorical_step(ret)
        else:
            raise NotImplementedError

        self._augment_observation(obs)
            
        info['vbf_average_posterior_belief_log_prob'] = meannpitem(self.previous_average_belief.log_prob(self.current_belief.sample()))
        return (obs, reward, done, info)
            
    def _gaussian_step(self, ret):
        # Compute the updated marginal terms.
        n = len(self.wrapper_bfs)
        latent_dist_n = self.wrapper_bfs.posterior_latent_dists[-1]
        dist_n = latent_dist_n.belief_distribution
        
        # dist_n = latent_dist_n.distribution_detached

        # This is to cover a specific case when the distribution isn't averaged over
        # dist_n.mean can't be modified
        dist_n_mean = dist_n.mean
        dist_n_variance = dist_n.variance
        if dist_n_mean.ndim == 3 and dist_n_mean.shape[0] == 1 and dist_n_mean.shape[1] == 1:
            dist_n_mean = dist_n_mean.squeeze()[None, :]
            dist_n_variance = dist_n_variance.squeeze()[None, :]

        assert(latent_dist_n.B == 1)
        assert(n - 1 == len(self._mean_sqs))
        assert(dist_n_mean.ndim == 2)
        assert(dist_n_mean.shape[0] == 1)
        assert(dist_n_variance.ndim == 2)
        assert(dist_n_variance.shape[0] == 1)
        assert(len(self._cov_diags) == len(self._mean_sqs))

        # Track state for moment matching.
        self._marginal_mean_accumulation += dist_n_mean
        self._mean_sqs.append(torch.pow(dist_n_mean, 2))
        self._cov_diags.append(dist_n_variance)

        assert(n == len(self._mean_sqs))
        self._set_avg_belief_from_gaussian_moment_match()
        return ret

    def _set_avg_belief_from_gaussian_moment_match(self):
        # Compute moment-matching parameters. See https://en.wikipedia.org/wiki/Mixture_distribution#Moments
        # Uniform mixture mean.
        n = len(self._mean_sqs)
        marginal_mean = self._marginal_mean_accumulation / n
        marginal_mean_sq = torch.pow(marginal_mean, 2)
        marginal_covariance_terms = [
            variance + mean_sq - marginal_mean_sq for (mean_sq, variance) in zip(self._mean_sqs, self._cov_diags)
        ]
        # Uniform mixture stddev.
        marginal_covariance_sqrt = torch.sqrt(sum(marginal_covariance_terms) / len(marginal_covariance_terms))

        self._avg_mean = marginal_mean
        self._avg_cov_sqrt = marginal_covariance_sqrt
        self._avg_suff_stats = torch.cat((self._avg_mean, self._avg_cov_sqrt), -1)
        self._avg_belief_dist = vbf.NormalLatentStateDistribution(
            marginal_mean, marginal_covariance_sqrt, mixture=False, do_build_rsample=False).distribution_detached

    @property
    def average_belief(self):
        return self._avg_belief_dist

    @property
    def previous_average_belief(self):
        return self._previous_avg_belief_dist    

    @property
    def current_belief(self):
        return self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution

def warn_present(kwargs):
    for k, v in kwargs.items():
        log.warning(f"Extra kwarg for wrapper: {k}={v}!")

class KLPosteriorAverageCostWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env, dimensionality_normalize=True, **kwargs):
        """
        :param env: 

        """
        super().__init__(env)
        warn_present(kwargs)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            belief = self.current_belief
            kl = float(vbf.kl_divergence(belief, self.average_belief).mean())
            kl_normalize = kl if not self.dimensionality_normalize else kl / np.prod(belief.event_shape)
            info['kl_posterior_average'] = kl_normalize
            assert('original_reward' not in info)
            info['original_reward'] = reward
            # NB this is negative --> "stabilize our current belief about the average belief"
            reward = -kl_normalize
            return dict_obs, reward, done, info

class KLPosteriorAverageRewardWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env, dimensionality_normalize=True):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            belief = self.current_belief
            kl = float(vbf.kl_divergence(belief, self.average_belief).mean())
            kl_normalize = kl if not self.dimensionality_normalize else kl / np.prod(belief.event_shape)
            info['kl_posterior_average'] = kl_normalize
            assert('original_reward' not in info)
            info['original_reward'] = reward
            # NB this is negative --> "stabilize our current belief about the average belief"
            reward = kl_normalize
            return dict_obs, reward, done, info

class PosteriorEntropyCostWrapper(Wrapper):
    
    @classu.member_initialize
    def __init__(self, env, use_ensemble_uncertainty=False):
        """

        :param env: 

        """
        super().__init__(env)
        if self.use_ensemble_uncertainty:
            raise NotImplementedError("this mode is deprecated -- use the wrapper instead")        
    
    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        
        
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = self.env.step(action)
        
        # NB this is not the belief, it is a sample!
        posterior = self.wrapper_bfs.posterior_latent_dists[-1].distribution

        posterior_entropy = info['vbf_posterior_entropy'] = meannpitem(vbf.entropy(posterior))
        assert('original_reward' not in info)
        info['original_reward'] = reward

        # Reduce posterior entropy
        reward = -1 * posterior_entropy
        return dict_obs, reward, done, info

class AveragePosteriorEntropyCostWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)

        # No need to convert to torch -- use the internal scipy dist of the marginal.
        average_entropy = meannpitem(vbf.entropy(self.average_belief))
        info['h_average'] = average_entropy
        assert('original_reward' not in info)
        info['original_reward'] = reward

        # NB this is negative --> "concentrate our average belief"
        reward = -average_entropy

        return dict_obs, reward, done, info

class VBFBeliefKLMixin:
    def compute_vbf_belief_kls(self, info, use_true_beliefs=True):
        if use_true_beliefs:
            posterior = self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution
            prior = self.wrapper_bfs.prior_latent_dists[-1].belief_distribution
            key = 'vbf_kl_posterior_belief_prior_belief'
        else:
            # NB these are not the beliefs! they're samples
            posterior = self.wrapper_bfs.posterior_latent_dists[-1].distribution
            prior = self.wrapper_bfs.prior_latent_dists[-1].distribution
            key = 'vbf_kl_posterior_prior'            
        kl_diff = meannpitem(vbf.kl_divergence(posterior, prior))
        info[key] = kl_diff
        info[key.replace('kl', 'xe')] = kl_diff + meannpitem(vbf.entropy(posterior))
            
class KLPosteriorPriorCostWrapper(gym.Wrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        self.compute_vbf_belief_kls(info)
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward = -info['vbf_kl_posterior_belief_prior_belief']
        return dict_obs, reward, done, info       

class KLPosteriorPriorRewardWrapper(gym.Wrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        self.compute_vbf_belief_kls(info)
        assert('original_reward' not in info)
        info['original_reward'] = reward        
        reward = info['vbf_kl_posterior_belief_prior_belief']
        return dict_obs, reward, done, info       


class EpistemicUncertaintyRewardWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward = info['vbf_prediction_variance']
        return dict_obs, reward, done, info

class JointInfogainRewardWrapper(gym.Wrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        self.compute_vbf_belief_kls(info)
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward0 = info['vbf_kl_posterior_belief_prior_belief']
        reward1 = info['vbf_prediction_variance']
        reward = reward0 + reward1
        return dict_obs, reward, done, info       
    

class EpistemicUncertaintyBonusWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env, dry=False, beta=1.0):
        """
        :param env: 

        """
        log.info(f"Creating epistemic bonus wrapper. Dry={self.dry}")
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        info['pre_bonus_reward'] = reward
        bonus = self.beta * np.sqrt(info['vbf_prediction_variance'])
        info['uncertainty_bonus'] = bonus
        
        # NB += here.
        if not self.dry:
            reward += bonus
        return dict_obs, reward, done, info    

class ZeroRewardWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward = 0.0
        return dict_obs, reward, done, info       

class EpistemicUncertaintyCostWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        assert('original_reward' not in info)
        info['original_reward'] = reward
        reward = -1 * np.sqrt(info['vbf_prediction_variance'])
        return dict_obs, reward, done, info

class InfogainAndKLPosteriorAverageCostWrapper(VBFAverageBeliefWrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            
            belief = self.current_belief
            # compute KL from belief to average
            kl_post_avg = float(vbf.kl_divergence(belief, self.average_belief).mean())
            info['vbf_kl_posterior_belief_average_belief'] = kl_post_avg

            # compute KL from belief to previous belief
            self.compute_vbf_belief_kls(info, use_true_beliefs=True)
            kl_post_prior = info['vbf_kl_posterior_belief_prior_belief']
            
            reward = kl_post_prior - kl_post_avg
            return dict_obs, reward, done, info

class StateInfogainRewardAndVisitationSurpriseCostWrapper(VBFAverageBeliefWrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env, dimensionality_normalize=True, split=False, infogain_multiplier=1.0, **kwargs):
        """
        :param env: 

        """
        super().__init__(env, **kwargs)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            
            # Compute infogain KL
            self.compute_vbf_belief_kls(info, use_true_beliefs=True)
            state_infogain_kl = info['vbf_kl_posterior_belief_prior_belief']
            # Compute KL to posterior
            xe_posterior_belief_average_belief = meannpitem(vbf.xe(self.current_belief, self.average_belief))
            info['vbf_xe_posterior_belief_average_belief'] = xe_posterior_belief_average_belief
            info['vbf_state_infogain_and_xe'] = self.infogain_multiplier * state_infogain_kl - xe_posterior_belief_average_belief

            if self.split:
                # TODO hack assuming T=100 
                if 0 <= self.env_timestep < 50:
                    reward = state_infogain_kl
                elif self.env_timestep == 50:
                    # Reset the average (final timestep before using it to stabilize)
                    self._average_reset()
                else:
                    reward = -1 * xe_posterior_belief_average_belief
            else:
                reward = info['vbf_state_infogain_and_xe']
                
            if self.dimensionality_normalize:
                reward /= np.prod(self.current_belief.event_shape)
            return dict_obs, reward, done, info        


class SplitModelInfogainAndKLPosteriorAverageCostWrapper(VBFAverageBeliefWrapper, VBFBeliefKLMixin):
    @classu.member_initialize
    def __init__(self, env, T, fraction=0.5):
        """
        :param env: 

        """
        super().__init__(env)
        self._T = T
        self._fraction = fraction
        assert(0 <= fraction <= 1)
        self._t_switch = int(round(self._T * fraction))
        if np.isclose(fraction, 1.) or np.isclose(fraction, 0.):
            log.warning(f"Switching time index will have no effect! (fraction={fraction})")

    def reset(self):
        self._t = 0
        return super().reset()
        
    def step(self, action):
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        if 'original_reward' not in info: info['original_reward'] = reward

        # Compute the KL cost
        kl_po_avg = float(vbf.kl_divergence(self.current_belief, self.average_belief).mean())
        info['vbf_kl_posterior_belief_average'] = kl_po_avg

        if self._t <= self._t_switch:
            # NB this is already computed in the VBFModelWrapper as long as the posterior is enabled.
            # Reward the policy for going to high-variance locations
            reward = np.sqrt(info['vbf_prediction_variance'])
        else:
            # Reward the policy for controlling the posterior.
            reward = -kl_po_avg

        self._t += 1
        return dict_obs, reward, done, info
    
class LatentStateInfogainRewardWrapper(VBFAverageBeliefWrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            
            # compute KL from belief to previous belief
            self.compute_vbf_belief_kls(info, use_true_beliefs=True)
            kl_post_prior = info['vbf_kl_posterior_belief_prior_belief']
            
            reward = kl_post_prior
            return dict_obs, reward, done, info

class AverageLatentStateInfogainRewardWrapper(VBFAverageBeliefWrapper, VBFBeliefKLMixin):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward

            prior = self.wrapper_bfs.prior_latent_dists[-1].belief_distribution
            kl_avg_pr = float(vbf.kl_divergence(self.average_belief, prior).mean())
            info['vbf_kl_average_belief_prior_belief'] = kl_avg_pr
            reward = kl_avg_pr
            return dict_obs, reward, done, info

class WorldModelObservationSurpriseRewardWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            assert(self.env.compute_surprise)
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            reward = info['vbf_surprise_t']
            return dict_obs, reward, done, info

class WorldModelObservationSurpriseCostWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            assert(self.env.compute_surprise)
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward
            reward = -1 * info['vbf_surprise_t']
            return dict_obs, reward, done, info

class LatentKLSurpriseCostWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env, parent_kwargs={}):
        """
        :param env: 

        """
        super().__init__(env, **parent_kwargs)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward

            # kl_po_prev_avg = float(vbf.kl_divergence(self.current_belief, self.previous_average_belief).mean())
            kl_po_avg = float(vbf.kl_divergence(self.current_belief, self.average_belief).mean())
            info['vbf_kl_posterior_belief_average_belief'] = kl_po_avg
            reward = -1 * kl_po_avg
            return dict_obs, reward, done, info

class ExpectedLatentVisitationSurpriseCostWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env, parent_kwargs={}):
        """
        :param env: 

        """
        super().__init__(env, **parent_kwargs)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward

            # kl_po_prev_avg = float(vbf.kl_divergence(self.current_belief, self.previous_average_belief).mean())
            belief_posterior = self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution
            kl_diff = meannpitem(vbf.kl_divergence(belief_posterior, self.average_belief))
            xe_diff = kl_diff + meannpitem(vbf.entropy(belief_posterior))
            expected_visitation_surprise = xe_diff
            info['vbf_expected_visitation_surprise'] = expected_visitation_surprise
            reward = -expected_visitation_surprise
            return dict_obs, reward, done, info

class LatentVisitationSurpriseCostWrapper(VBFAverageBeliefWrapper):

    @classu.member_initialize
    def __init__(self, env, dimensionality_normalize=True, **kwargs):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        with torch.no_grad():
            # Step the chosen action in the environment.
            dict_obs, reward, done, info = super().step(action)
            info['original_reward'] = reward

            # kl_po_prev_avg = float(vbf.kl_divergence(self.current_belief, self.previous_average_belief).mean())
            belief_posterior = self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution        
            log_qbar_z = meannpitem(self.average_belief.log_prob(belief_posterior.mean))

            score_normalize = log_qbar_z if not self.dimensionality_normalize else log_qbar_z / np.prod(belief_posterior.event_shape)
            
            info['vbf_log_avg_belief'] = log_qbar_z
            reward = score_normalize
            return dict_obs, reward, done, info

class StateEntropyInfo(gym.Wrapper):
    def reset(self):
        self._welford_continuous = welford.Welford()
        self._welford_discrete = welford.Welford()
        self._welford_env = welford.Welford()        
        dict_obs = super().reset()
        self.visitation = collections.defaultdict(lambda: 0)
        self._update_dists(dict_obs)
        return dict_obs

    def _update_dists(self, dict_obs):
        if dict_obs is None: return dict_obs
        if 'state' in dict_obs:
            state = dict_obs['state']
            self._welford_continuous.add(state)
        if 'discrete_state' in dict_obs:
            # TODO if the state's discrete, should use something smarter than a normal to approximate the marginal
            discrete_state = dict_obs['discrete_state']
            self._welford_discrete.add(discrete_state)
        # if 'env_state' in dict_obs:
        #     env_state = dict_obs['env_state']
        #     self._welford_env.add(env_state)            
        return dict_obs

    def get_visitation(self):
        return self.visitation

    def set_visitation(self, x):
        self.visitation = x
            
    def step(self, action):
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        self._update_dists(dict_obs)
        eps = 1e-4

        if 'state' in dict_obs:
            state = dict_obs['state']
            # Entropy of MVN = .5 ln((2pie)^k * det(Sigma))
            det = np.prod(self._welford_continuous.var_s + eps)
            k = len(state)
            c_entropy = .5 * (k * np.log(2 * np.pi) + np.log(det))
            info['differential_entropy_continuous_state_visitation'] = c_entropy
        if 'discrete_state' in dict_obs:
            # TODO if the state's discrete, should use something smarter than a normal to approximate the marginal
            discrete_state = dict_obs['discrete_state']
            # Entropy of MVN = .5 ln((2pie)^k * det(Sigma))
            det = np.prod(self._welford_discrete.var_s + eps)
            k = len(discrete_state)
            d_entropy = .5 * (k * np.log(2 * np.pi) + np.log(det))
            info['differential_entropy_continuous_discrete_state_visitation'] = d_entropy
        if 'env_state' in dict_obs:
            # TODO if the state's discrete, should use something smarter than a normal to approximate the marginal
            env_state = dict_obs['env_state']
            # # Entropy of MVN = .5 ln((2pie)^k * det(Sigma))
            # det = np.prod(self._welford_env.var_s + eps)
            # k = len(env_state)
            # d_entropy = .5 * (k * np.log(2 * np.pi) + np.log(det))
            # info['differential_entropy_continuous_env_state_visitation'] = d_entropy
            if isinstance(env_state, np.ndarray):
                env_state = tuple(env_state.tolist())
            self.visitation[env_state] += 1
            # n_possible_states = np.prod(self.observation_space['env_state'].high - self.observation_space['env_state'].low)
            n_total_visits = sum(self.visitation.values())
            entropy_sum = 0
            for state, visits in self.visitation.items():
                p_i = visits / n_total_visits
                entropy_sum += - p_i * np.log(p_i)
            # TODO Dbg
            if np.isclose(entropy_sum, 0) and len(self.visitation) > 1:
                pdb.set_trace()
            info['discrete_entropy_env_state'] = entropy_sum
        return dict_obs, reward, done, info

class VBFEntropyInfo(VBFAverageBeliefWrapper, VBFBeliefKLMixin):
    def __init__(self, env, entropy_keys, parent_kwargs={}):
        super().__init__(env, **parent_kwargs)
        self.entropy_keys = entropy_keys        
        
    def step(self, action):
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)

        if 'original_reward' not in info:
            info['original_reward'] = reward

        if all(_.find('vbf') == -1 for _ in self.entropy_keys):
            return dict_obs, reward, done, info            

        # NB this is not the belief, it is a sample!
        posterior = self.wrapper_bfs.posterior_latent_dists[-1].distribution
        previous_posterior = self.wrapper_bfs.posterior_latent_dists[-2].distribution
        belief_posterior = self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution
        belief_prior = self.wrapper_bfs.prior_latent_dists[-1].belief_distribution
        prior = self.wrapper_bfs.prior_latent_dists[-1].distribution

        NM = np.prod(self.current_belief.event_shape)
            
        if 'vbf_kl_average_belief_posterior_belief' in self.entropy_keys:
            info['vbf_kl_average_belief_posterior_belief'] = float(vbf.kl_divergence(self.average_belief, self.current_belief).mean())
        if 'vbf_prior_entropy' in self.entropy_keys:
            info['vbf_prior_entropy'] = meannpitem(vbf.entropy(prior))
        if 'vbf_posterior_entropy' in self.entropy_keys:
            info['vbf_posterior_entropy'] = meannpitem(vbf.entropy(posterior))

        info['vbf_kl_posterior_belief_average'] = float(vbf.kl_divergence(self.current_belief, self.average_belief).mean())
        info['vbf_prior_entropy'] = meannpitem(vbf.entropy(prior))
        info['vbf_prior_belief_entropy'] = meannpitem(vbf.entropy(belief_prior))
        info['vbf_posterior_entropy'] = meannpitem(vbf.entropy(posterior))
        info['vbf_posterior_belief_entropy'] = meannpitem(vbf.entropy(belief_posterior))
        info['vbf_average_posterior_belief_entropy'] = meannpitem(vbf.entropy(self.average_belief))
        info['vbf_neg_kl_posterior_belief_average'] = -1 * info['vbf_kl_posterior_belief_average'] / 1
        info['vbf_certainty'] = -1 * info['vbf_posterior_entropy'] / 1
        
        kl_avg_pr = float(vbf.kl_divergence(self.average_belief, belief_prior).mean())

        kl_avg_prev_avg = float(vbf.kl_divergence(self.average_belief, self.previous_average_belief).mean())
        info['vbf_kl_average_belief_previous_average_belief'] = kl_avg_prev_avg
        
        kl_po_prev_avg = float(vbf.kl_divergence(self.current_belief, self.previous_average_belief).mean())
        info['vbf_kl_posterior_belief_previous_average_belief'] = kl_po_prev_avg
        
        kl_po_prev_po = float(vbf.kl_divergence(posterior, previous_posterior).mean())

        info['vbf_kl_average_belief_prior_belief'] = kl_avg_pr
        info['vbf_kl_posterior_previous_posterior'] = kl_po_prev_po

        # Compute both.
        self.compute_vbf_belief_kls(info, use_true_beliefs=True)
        self.compute_vbf_belief_kls(info, use_true_beliefs=False)

        # Combination terms.
        info['vbf_kl0_kl1'] = info['vbf_kl_posterior_belief_prior_belief'] - info['vbf_kl_posterior_belief_average']
        info['vbf_kl0_h0'] = info['vbf_kl_posterior_belief_prior_belief'] - info['vbf_average_posterior_belief_entropy']

        belief_posterior = self.wrapper_bfs.posterior_latent_dists[-1].belief_distribution
        
        # pdb.set_trace()
        # log_qbar_z = meannpitem(self.average_belief.log_prob(belief_posterior.sample()))
        log_qbar_z = meannpitem(self.average_belief.log_prob(belief_posterior.mean))
        info['vbf_log_avg_belief'] = log_qbar_z
        info['vbf_kl0_qb'] = (info['vbf_kl_posterior_belief_prior_belief'] + info['vbf_log_avg_belief']) / NM

        xe_posterior_belief_average_belief = meannpitem(vbf.xe(self.current_belief, self.average_belief))

        t0 = info['vbf_kl_posterior_belief_prior_belief'] / NM
        t1 = - xe_posterior_belief_average_belief/ NM
        info['vbf_infogain_and_xe'] = t0 + t1
        info['vbf_infogain'] = info['vbf_kl_posterior_belief_prior_belief']

        if 0 <= self.env_timestep <= 50:
            # info['vbf_infogain_and_xe'] = t0
            info['vbf_split'] = info['vbf_kl_posterior_belief_prior_belief']
        else:
            info['vbf_split'] = -1 * info['vbf_average_posterior_belief_entropy']
            # info['vbf_infogain_and_xe'] = t1

        # if self.env_timestep == 50: self._average_reset()
            
        return dict_obs, reward, done, info
    
class EERewardWrapper(gym.Wrapper):

    def __init__(self, env, eeModel):
        super().__init__(env)
        self._model = eeModel

    def step(self, action):
        import cv2
        obs, env_rew, envdone, info = super().step(action)
        info["env_rew"] = env_rew
        o2 = np.transpose(obs["image"], (1, 2, 0))
        o3 = cv2.resize(o2, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        env_rew = self._model.water_filling_from_observations([o3], batch_size=1).item()
        return obs, env_rew, envdone, info

class PreprocessDiscreteAction(gym.Wrapper):
    def step(self, action):
        if isinstance(action, np.ndarray):
            assert(action.size == 1)
            action = action.item()
        assert(isinstance(action, int))
        return super().step(action)

class ExponentialRewardWrapper(gym.Wrapper):

    @classu.member_initialize
    def __init__(self, env):
        """
        :param env: 

        """
        super().__init__(env)

    def step(self, action):
        """

        NB we override the 'step' function because the updated reward is a function of the environment info.

        """
        # Step the chosen action in the environment.
        dict_obs, reward, done, info = super().step(action)
        if 'original_reward' not in info:
            info['original_reward'] = reward
        reward = np.exp(reward)
        return dict_obs, reward, done, info
