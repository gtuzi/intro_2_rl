from typing import Union, Callable, Any, Optional, Tuple
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.tensor.parallel import loss_parallel
from torch.utils.tensorboard import SummaryWriter

from shared.utils import (
    NoiseSchedule,
    Experience,
    NoiseSchedule,
    SoftPolicy
)

from policy_gradient.utils import (
    DiscreteActionAgent,
    ContinuousActionAgent,
    DiscreteActionSoftPolicy, DiscreteActionCriticStateValue,
    ContinuousActionSoftPolicy, ContinuousActionCriticStateValue
)

from policy_gradient.nets import (
    DiscreteActionPolicyMLP,
    ValueFunction,
    GaussianPolicy
)

BIG_NUMBER = 1e8
SMALL_NUMBER = 1e-8

def softmax(logits: np.ndarray, temp: float = 1.):
    assert 0 < logits.ndim < 3
    assert isinstance(temp, float)

    eaxis = None

    if logits.ndim == 2:
        assert 1 in logits.shape
        eaxis = int(logits.shape[1] == 1)
        logits = logits.squeeze()
    m = np.max(logits)
    z = (logits - m) / temp
    res = np.exp(z) / (np.sum(np.exp(z)) + SMALL_NUMBER)

    return np.expand_dims(res, axis=eaxis) if eaxis is not None else res


def log_softmax(logits, temp=1.0):
    z = logits / temp
    z_max = np.max(z)                      # for numerical stability
    shifted = z - z_max
    lse = z_max + np.log(np.sum(np.exp(shifted)))
    return z - lse


class Reinforce_LinearApproximation(DiscreteActionSoftPolicy):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            policy_feature_fn: Callable[[Any, ], np.ndarray], # state --> np.ndarray
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, NoiseSchedule] = 1.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, NoiseSchedule)

        super().__init__(
            state_size=feature_size,
            action_space_dims=action_space_dims,
            discount=discount,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.update_coefficient = update_coefficient
        self.buffer = []
        self.w = None
        self.policy_feature_fn = policy_feature_fn
        self.init_weights()

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    def init_weights(self, *args, **kwargs):
        if 'init' in kwargs:
            self.w = kwargs['init']((self.action_space_dims, self.feature_size))
        else:
            self.w = 0.01 * np.random.uniform(size=(self.action_space_dims, self.feature_size)).astype(np.float32)
            # self.w = np.zeros((self.action_space_dims, self.feature_size), dtype=np.float32) #[features|actions]

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.initialize()

        self.init_weights()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.temp, NoiseSchedule):
            self.temp.reset()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.reset()

        self.buffer = []

    def act(self, state) -> Tuple[int, float]:
        x = self.policy_feature_fn(state)
        acts = list(range(self.action_space_dims))
        logits = np.matmul(self.w, x)

        t = (
            self.temp.value
            if isinstance(self.temp,NoiseSchedule)
            else self.temp
        )

        probs = softmax(logits, t)

        if probs.ndim > 1:
            probs = probs.squeeze()

        a = self.rng.random.choice(acts, replace=True, p=probs)

        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        return a, probs[a]

    def get_sa_probability(self, s, a):
        x = self.policy_feature_fn(s)
        logits = np.matmul(self.w, x)

        t = (
            self.temp.value
            if isinstance(self.temp, NoiseSchedule)
            else self.temp
        )

        return softmax(logits, t)[a]

    def get_greedy_action(self, s):
        """
            Get the greedy action and its **conditional** prob.
            If a single action: conditional probability is 1.
            If multiple actions compete for being picked, they are randomly
            tie-broken. This mean's that their probability is 1/|argmax_a|
        """

        x = self.policy_feature_fn(s)
        logits = np.matmul(self.w, x)

        if logits.ndim > 1:
            logits = logits.squeeze()

        max_vals = np.amax(logits)
        idc = np.argwhere(logits == max_vals).squeeze().tolist()

        if isinstance(idc, list):
            # Random tie-breaking
            return int(self.rng.random.choice(idc)), 1. / len(idc)
        else:
            assert isinstance(idc, int)
            return idc, 1.

    def logp(self, s):
        x = self.policy_feature_fn(s)
        logits = np.matmul(self.w, x)

        t = (
            self.temp.value
            if isinstance(self.temp, NoiseSchedule)
            else self.temp
        )

        return log_softmax(logits, t)

    def entropy(self, s):
        logp = self.logp(s)
        return -np.sum([np.exp(lp)*lp for lp in logp])

    def grad_logpi_w(self, a, s):
        """
            d_p[a] / d_w[j] = p[a](δ_{aj} - p[j])
            grad(log(pi)) = (δ_{aj} - p[j])
            * -p[j] if a != j
            * 1 - p[j] if a == j
        """
        p = np.exp(self.logp(s))
        p = -p.squeeze()
        p[a] += 1.
        x = self.policy_feature_fn(s)
        return np.array([p[_a] * x for _a in range(self.action_space_dims)])[..., 0]

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))
        if experience.done:
            self._learn()
            self.buffer.clear()
        self.t += 1

    def _learn(self):

        T = len(self.buffer)
        Gs = np.zeros((T, ))

        def _G_update(i, r, done):
            Gs[i] = r if done else r + self.discount * Gs[i+1]

        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        if isinstance(self.update_coefficient, NoiseSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        else:
            alpha = self.update_coefficient

        for experience, t in self.buffer:
            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            weight_update = alpha * (
                    self.discount ** t) * Gs[t] * self.grad_logpi_w(a, s)
            self.w += weight_update

        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

# ======================================================================== #

def to_tensor(s, dtype=None, device=None):
    if isinstance(s, torch.Tensor):
        pass
    elif not isinstance(s, np.ndarray):
        assert dtype is not None
        s = torch.tensor(np.array(s), dtype=dtype)
    else:
        s = torch.from_numpy(s)

    if device is not None:
        s = s.to(device)

    return s


def to_native(s: Union[torch.Tensor, np.ndarray]):

    if isinstance(s, torch.Tensor):
        s = s.detach().cpu().numpy()

    if isinstance(s, np.ndarray):
        dtype = s.dtype

        if s.size == 1:
            if dtype in (np.float32, np.float64):
                return float(s)
            elif dtype in (np.int32, np.int64):
                return int(s)
        else:
            return s.tolist()

    return s


def to_tensor_state_action(s, a, device=None):
    if isinstance(a, np.ndarray):
        assert a.dtype in (np.int32, np.int64)
        assert isinstance(s, np.ndarray)
        assert a.shape[0] == s.shape[0]
        s = to_tensor(s)
    elif isinstance(a, int):
        a = to_tensor([a], torch.long)
        assert not isinstance(s, np.ndarray)
        s = to_tensor(s, dtype=torch.float32)
    elif isinstance(a, (tuple, list)):
        s = to_tensor(s, dtype=torch.float32)
        is_float = isinstance(a[0], float)
        a = to_tensor(a, dtype=torch.float32 if is_float else None)
    elif isinstance(a, torch.Tensor):
        assert isinstance(s, torch.Tensor)
    else:
        raise Exception("Input types not recognized")

    if device is not None:
        s = s.to(device)
        a = a.to(device)

    return s, a


####################################################
############### Discrete Action ####################
####################################################


class Reinforce(DiscreteActionSoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, NoiseSchedule] = 1.,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            normalize_reward: bool = False,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, NoiseSchedule)

        assert norm_threshold > 0.

        super().__init__(
            state_size=state_size,
            action_space_dims=action_space_dims,
            discount=discount,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient = update_coefficient
        self.buffer = []
        self.policy = None
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold
        self.normalize_reward = normalize_reward

    def init_model(self, *args, **kwargs):
        self.policy = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.initialize()

        self.init_model()
        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.buffer = []

    def act(self, state, native = True) -> Tuple[int, float]:
        with torch.no_grad():
            actions, probs = self.policy.sample(
                to_tensor(state, dtype=torch.float32),
                temperature=self.temperature,
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)

        return actions, probs

    def get_sa_probability(self, s, a, native = True):
        s, a = to_tensor_state_action(s, a)
        p = self.policy.prob_sa(s, a, self.temperature)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native = True):
        with torch.no_grad():
            probs = self.policy.prob_s(
                to_tensor(s, dtype=torch.float32),
                temperature=self.temperature
            )

        actions = probs.argmax(dim=-1)
        probs = probs[actions, ...]

        if native:
            actions = to_native(actions)
            probs = to_native(probs)

        return actions, probs

    def logp(self, s, native = True):
        res = self.policy.logprob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(res) if native else res

    def prob(self, s, native = True):
        probs = self.policy.prob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(probs) if native else probs

    def logp_sa(self, s, a, native=True):
        res = self.policy.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

    def entropy(self, s, native = True):
        e = self.policy.entropy(
            to_tensor(s, torch.float32),
            temperature=self.temperature
        )

        return to_native(e) if native else e

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))

        loss = 0
        # MC - we learn at the end of the episode
        if experience.done:
            loss = self._learn()
            self.buffer.clear()
            self.reset()
        else:
            self.t += 1

        return loss

    def _learn(self):

        if isinstance(self.update_coefficient, NoiseSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        else:
            alpha = self.update_coefficient

        T = len(self.buffer)

        Gs = np.zeros((T, ))

        R = [e.r for e, _ in self.buffer] if self.normalize_reward else 1.
        R, sig = (np.mean(R), np.std(R)) if self.normalize_reward else (0., 1.)

        def _G_update(i, r, done):
            nr = (r - R)/(sig + 1e-8)
            Gs[i] = nr if done else nr + self.discount * Gs[i+1]

        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        avg_grad_norm = 0.

        for experience, t in self.buffer:

            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            logp = self.logp_sa(s, [a], native=False)
            assert not torch.any(torch.isnan(logp))

            weights = list(self.policy.parameters())
            grads = torch.autograd.grad(logp, weights, retain_graph=False)

            flat_grads = [g.flatten() for g in grads]
            gradient_norm = float(torch.norm(torch.cat(flat_grads)))
            avg_grad_norm += gradient_norm

            do_norm = gradient_norm > self.norm_threshold and self.norm_grad
            k = self.norm_threshold / gradient_norm if do_norm else 1.

            with torch.no_grad():
                for w, g in zip(weights, grads):
                    update = alpha * (self.discount ** t) * Gs[t]
                    update *= g * k
                    w += update

        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        if len(self.buffer) > 0:
            return avg_grad_norm / len(self.buffer)
        else:
            return avg_grad_norm


class ReinforceBaseline(
    DiscreteActionSoftPolicy,
    DiscreteActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            update_coefficient_critic: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, NoiseSchedule] = 1.,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            normalize_reward: bool = False,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        if norm_threshold is not None:
            assert norm_threshold > 0.

        super().__init__(
            state_size=state_size,
            action_space_dims=action_space_dims,
            discount=discount,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_actor
        self.update_coefficient_baseline = update_coefficient_critic
        self.buffer = []
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold
        self.normalize_reward = normalize_reward

    def init_model(self, *args, **kwargs):
        self.policy = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

        # Using policy as feature extractor. One feature per action
        self.value = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_baseline, NoiseSchedule):
            self.update_coefficient_baseline.initialize()

        self.init_model()

        self.buffer = []

    def reset(self):
        # Preparing for a new episode
        self.t = 0
        self.buffer = []

    def act(self, state, native=True) -> Tuple[int, float]:
        with torch.no_grad():
            actions, probs = self.policy.sample(
                to_tensor(state, dtype=torch.float32),
                temperature=self.temperature,
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)

        return actions, probs

    def get_sa_probability(self, s, a, native=True):
        s, a = to_tensor_state_action(s, a)
        p = self.policy.prob_sa(s, a, self.temperature)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native=True):
        with torch.no_grad():
            probs = self.policy.prob_s(
                to_tensor(s, dtype=torch.float32),
                temperature=self.temperature
            )

        actions = probs.argmax(dim=-1)
        probs = probs[actions, ...]

        if native:
            actions = to_native(actions)
            probs = to_native(probs)

        return actions, probs

    def logp(self, s, native=True):
        res = self.policy.logprob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(res) if native else res

    def prob(self, s, native=True):
        probs = self.policy.prob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(probs) if native else probs

    def logp_sa(self, s, a, native=True):
        res = self.policy.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

    def entropy(self, s, native=True):
        e = self.policy.entropy(
            to_tensor(s, torch.float32),
            temperature=self.temperature
        )

        return to_native(e) if native else e

    def state_value(self, s, native=True, **kwargs) -> float:
        v = self.value(to_tensor(s, dtype=torch.float32))
        return to_native(v) if native else v

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))

        loss = 0
        # MC - we learn at the end of the episode
        if experience.done:
            loss = self._learn()
            self.buffer.clear()
            self.reset()
        else:
            self.t += 1

        return loss

    def _learn(self):
        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            alpha_pi = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_pi = self.update_coefficient_policy

        if isinstance(self.update_coefficient_baseline, NoiseSchedule):
            alpha_b = self.update_coefficient_baseline.value
            self.update_coefficient_baseline.step()
        else:
            alpha_b = self.update_coefficient_baseline


        R = [e.r for e, _ in self.buffer] if self.normalize_reward else 1.
        R, sig = (np.mean(R), np.std(R)) if self.normalize_reward else (0., 1.)

        T = len(self.buffer)
        Gs = np.zeros((T,))

        def _G_update(i, r, done):
            nr = (r - R) / (sig + 1e-8)
            Gs[i] = nr if done else nr + self.discount * Gs[i + 1]

        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        loss = 0.
        for experience, t in self.buffer:
            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            # --- Update baseline --- #
            v = self.value(to_tensor(s, dtype=torch.float32))
            delta = Gs[t] - v
            loss += delta

            W = list(self.value.parameters())
            grads_v = torch.autograd.grad(v, W, retain_graph=False)

            flat_grads_v = [g.flatten() for g in grads_v]
            gradient_norm_v = float(torch.norm(torch.cat(flat_grads_v)))

            do_norm_v = gradient_norm_v > self.norm_threshold and self.norm_grad
            kv = self.norm_threshold / gradient_norm_v if do_norm_v else 1.

            with torch.no_grad():
                for w, g in zip(W, grads_v):
                    update = alpha_b * delta * (g * kv)
                    w += update

            # --- Update Actor --- #
            logp = self.logp_sa(s, [a], native=False)
            assert not torch.any(torch.isnan(logp))
            theta = list(self.policy.parameters())
            grads_pi = torch.autograd.grad(logp, theta, retain_graph=False)

            flat_grads_pi = [g.flatten() for g in grads_pi]
            gradient_norm_pi = float(torch.norm(torch.cat(flat_grads_pi)))

            do_norm_pi = gradient_norm_pi > self.norm_threshold and self.norm_grad
            kpi = self.norm_threshold / gradient_norm_pi if do_norm_pi else 1.

            with torch.no_grad():
                for th, g in zip(theta, grads_pi):
                    update = alpha_pi * (self.discount ** t) * delta * (g * kpi)
                    th += update

        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        return loss / len(self.buffer) if len(self.buffer) > 0 else loss


class OneStepAC(
    DiscreteActionSoftPolicy,
    DiscreteActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            update_coefficient_critic: Union[float, NoiseSchedule],
            hidden_dims=(32,),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, NoiseSchedule] = 1.,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        super().__init__(
            state_size=state_size,
            action_space_dims=action_space_dims,
            discount=discount,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_actor
        self.update_coefficient_critic = update_coefficient_critic
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold
        self.I = 1

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            self.update_coefficient_critic.initialize()

        self.init_model()
        self.I = 1
        self.t = 0

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.I = 1

    def get_sa_probability(self, s, a, native=True):
        s, a = to_tensor_state_action(s, a)
        p = self.actor.prob_sa(s, a, self.temperature)
        return to_native(p) if native else p

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

    def logp(self, s, native=True):
        res = self.actor.logprob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(res) if native else res

    def prob(self, s, native=True):
        probs = self.actor.prob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(probs) if native else probs

    def entropy(self, s, native=True, **kwargs):
        e = self.actor.entropy(
            to_tensor(s, torch.float32),
            temperature=self.temperature
        )

        return to_native(e) if native else e

    def state_value(self, s,  native=True, **kwargs) -> float:
        v = self.critic(to_tensor(s, dtype=torch.float32))
        return to_native(v) if native else v

    def act(self, state, native=True) -> Tuple[int, float]:
        with torch.no_grad():
            actions, probs = self.actor.sample(
                to_tensor(state, dtype=torch.float32),
                temperature=self.temperature,
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)

        return actions, probs

    def get_greedy_action(self, s, native=True):
        with torch.no_grad():
            probs = self.actor.prob_s(
                to_tensor(s, dtype=torch.float32),
                temperature=self.temperature
            )

        actions = probs.argmax(dim=-1)
        probs = probs[actions, ...]

        if native:
            actions = to_native(actions)
            probs = to_native(probs)

        return actions, probs

    def step(self, experience: Experience, **kwargs) -> float:
        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            alpha_critic = self.update_coefficient_critic.value
            self.update_coefficient_critic.step()
        else:
            alpha_critic = self.update_coefficient_critic

        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        avg_grad_norm_v = 0.
        avg_grad_norm_pi = 0.

        v = self.critic(to_tensor(s))

        with torch.no_grad():
            vp = self.critic(to_tensor(sp)) * (1 - done)
            delta = (r + self.discount * vp) - v

        # --- Update critic --- #
        W = list(self.critic.parameters())
        grads_v = torch.autograd.grad(v, W, retain_graph=False)

        flat_grads_v = [g.flatten() for g in grads_v]
        gradient_norm_v = float(torch.norm(torch.cat(flat_grads_v)))
        avg_grad_norm_v += gradient_norm_v

        do_norm_v = gradient_norm_v > self.norm_threshold and self.norm_grad
        kv = self.norm_threshold / gradient_norm_v if do_norm_v else 1.

        with torch.no_grad():
            for w, g in zip(W, grads_v):
                update = alpha_critic * delta * (g * kv)
                w += update

        # --- Update Actor --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        theta = list(self.actor.parameters())
        grads_pi = torch.autograd.grad(logp, theta, retain_graph=False)

        flat_grads_pi = [g.flatten() for g in grads_pi]
        gradient_norm_pi = float(torch.norm(torch.cat(flat_grads_pi)))
        avg_grad_norm_pi += gradient_norm_pi

        do_norm_pi = gradient_norm_pi > self.norm_threshold and self.norm_grad
        kpi = self.norm_threshold / gradient_norm_pi if do_norm_pi else 1.

        with torch.no_grad():
            for th, g in zip(theta, grads_pi):
                update = alpha_actor * self.I * delta * (g * kpi)
                th += update

        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        self.I *= self.discount
        self.t += 1

        return to_native(delta)


class ACWithEligibilityTraces(
    DiscreteActionSoftPolicy,
    DiscreteActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            lam_actor: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            hidden_dims=(32,),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, NoiseSchedule] = 1.,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        assert 0 <= lam_actor <= 1
        assert 0 <= lam_critic <= 1

        super().__init__(
            state_size=state_size,
            action_space_dims=action_space_dims,
            discount=discount,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_actor
        self.update_coefficient_critic = update_coefficient_critic
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold
        self.I = 1
        self.z_critic = None
        self.z_actor = None

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            self.update_coefficient_critic.initialize()

        self.init_model()
        self.I = 1
        self.t = 0
        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.I = 1
        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

    def get_sa_probability(self, s, a, native=True):
        s, a = to_tensor_state_action(s, a)
        p = self.actor.prob_sa(s, a, self.temperature)
        return to_native(p) if native else p

    def logp(self, s, native=True):
        res = self.actor.logprob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(res) if native else res

    def prob(self, s, native=True):
        probs = self.actor.prob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(probs) if native else probs

    def state_value(self, s,  native=True, **kwargs) -> float:
        v = self.critic(to_tensor(s, dtype=torch.float32))
        return to_native(v) if native else v

    def entropy(self, s, native=True):
        e = self.actor.entropy(
            to_tensor(s, torch.float32),
            temperature=self.temperature
        )

        return to_native(e) if native else e

    def act(self, state, native=True) -> Tuple[int, float]:
        with torch.no_grad():
            actions, probs = self.actor.sample(
                to_tensor(state, dtype=torch.float32),
                temperature=self.temperature,
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)

        return actions, probs

    def get_greedy_action(self, s, native=True):
        with torch.no_grad():
            probs = self.actor.prob_s(
                to_tensor(s, dtype=torch.float32),
                temperature=self.temperature
            )

        actions = probs.argmax(dim=-1)
        probs = probs[actions, ...]

        if native:
            actions = to_native(actions)
            probs = to_native(probs)

        return actions, probs

    def step(self, experience: Experience, **kwargs) -> float:
        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            alpha_critic = self.update_coefficient_critic.value
            self.update_coefficient_critic.step()
        else:
            alpha_critic = self.update_coefficient_critic

        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        v = self.critic(to_tensor(s))

        with torch.no_grad():
            vp = self.critic(to_tensor(sp)) * (1 - done)
            delta = (r + self.discount * vp) - v

        # --- Update critic ETs --- #
        grads_v = torch.autograd.grad(
            v,
            list(self.critic.parameters()),
            retain_graph=False
        )

        flat_grads_v = [g.flatten() for g in grads_v]
        gradient_norm_v = float(torch.norm(torch.cat(flat_grads_v)))

        do_norm_v = gradient_norm_v > self.norm_threshold and self.norm_grad
        kv = self.norm_threshold / gradient_norm_v if do_norm_v else 1.

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i] = (
                        self.discount * self.lam_critic * self.z_critic[i] +
                        (grads_v[i] * kv)
                )

        # --- Update Actor ETs --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        grads_pi = torch.autograd.grad(
            logp,
            list(self.actor.parameters()),
            retain_graph=False
        )

        flat_grads_pi = [g.flatten() for g in grads_pi]
        gradient_norm_pi = float(torch.norm(torch.cat(flat_grads_pi)))

        do_norm_pi = gradient_norm_pi > self.norm_threshold and self.norm_grad
        kpi = self.norm_threshold / gradient_norm_pi if do_norm_pi else 1.

        with torch.no_grad():
            for i in range(len(self.z_actor)):
                self.z_actor[i] = (
                        self.discount * self.lam_actor * self.z_actor[i] +
                        self.I * (grads_pi[i] * kpi)
                )

        # ---- Update Critic --- #
        with torch.no_grad():
            for z, w in zip(self.z_critic, self.critic.parameters()):
                w += alpha_critic * delta * z

        # ---- Update Actor ---- #
        with torch.no_grad():
            for z, th in zip(self.z_actor, self.actor.parameters()):
                th += alpha_actor * delta * z


        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        self.I *= self.discount
        self.t += 1

        return to_native(delta)


class ACWithEligibilityTracesContinuing(
    DiscreteActionSoftPolicy,
    DiscreteActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            lam_actor: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            update_coefficient_avg_reward: Union[float, NoiseSchedule],
            hidden_dims=(32,),
            temp: Union[float, NoiseSchedule] = 1.,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            seed: Optional[int] = None
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        if isinstance(update_coefficient_avg_reward, float):
            assert 0. < update_coefficient_avg_reward < 1.
        else:
            assert isinstance(update_coefficient_avg_reward, NoiseSchedule)

        assert 0 <= lam_actor <= 1
        assert 0 <= lam_critic <= 1

        super().__init__(
            state_size=state_size,
            action_space_dims=action_space_dims,
            discount=0,
            temp=temp,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_actor
        self.update_coefficient_critic = update_coefficient_critic
        self.update_coefficient_avg_reward = update_coefficient_avg_reward
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self._writer: Optional[SummaryWriter] = None
        self.temp = temp
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold
        self.z_critic = None
        self.z_actor = None
        self.R_bar = 0.

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input,
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input,
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            self.update_coefficient_critic.initialize()

        if isinstance(self.update_coefficient_avg_reward, NoiseSchedule):
            self.update_coefficient_avg_reward.initialize()

        self.init_model()
        self.t = 0
        self.R_bar = 0

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def reset(self):
        # The agent here is prepared for a new episode, that forthe
        # continuing case should be done after a long long time.
        self.t = 0
        self.R_bar = 0

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

    def get_sa_probability(self, s, a, native=True):
        s, a = to_tensor_state_action(s, a)
        p = self.actor.prob_sa(s, a, self.temperature)
        return to_native(p) if native else p

    def logp(self, s, native=True):
        res = self.actor.logprob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(res) if native else res

    def prob(self, s, native=True):
        probs = self.actor.prob_s(
            to_tensor(s, dtype=torch.float32),
            temperature=self.temperature
        )

        return to_native(probs) if native else probs

    def state_value(self, s,  native=True, **kwargs) -> float:
        v = self.critic(to_tensor(s, dtype=torch.float32))
        return to_native(v) if native else v

    def entropy(self, s, native=True):
        e = self.actor.entropy(
            to_tensor(s, torch.float32),
            temperature=self.temperature
        )

        return to_native(e) if native else e

    def act(self, state, native=True) -> Tuple[int, float]:
        with torch.no_grad():
            actions, probs = self.actor.sample(
                to_tensor(state, dtype=torch.float32),
                temperature=self.temperature,
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)

        return actions, probs

    def get_greedy_action(self, s, native=True):
        with torch.no_grad():
            probs = self.actor.prob_s(
                to_tensor(s, dtype=torch.float32),
                temperature=self.temperature
            )

        actions = probs.argmax(dim=-1)
        probs = probs[actions, ...]

        if native:
            actions = to_native(actions)
            probs = to_native(probs)

        return actions, probs

    def step(self, experience: Experience, **kwargs):

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            alpha_critic = self.update_coefficient_critic.value
            self.update_coefficient_critic.step()
        else:
            alpha_critic = self.update_coefficient_critic

        if isinstance(self.update_coefficient_avg_reward, NoiseSchedule):
            alpha_reward = self.update_coefficient_avg_reward.value
            self.update_coefficient_avg_reward.step()
        else:
            alpha_reward = self.update_coefficient_avg_reward

        avg_grad_norm_v = 0.
        avg_grad_norm_pi = 0.

        s, a, r, sp, ap = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap
        )

        v = self.critic(to_tensor(s))

        with torch.no_grad():
            vp = self.critic(to_tensor(sp))
            delta = (r - self.R_bar + vp) - v

        self.R_bar += alpha_reward * delta

        # --- Update critic ETs --- #
        grads_v = torch.autograd.grad(
            v,
            list(self.critic.parameters()),
            retain_graph=False
        )

        flat_grads_v = [g.flatten() for g in grads_v]
        gradient_norm_v = float(torch.norm(torch.cat(flat_grads_v)))
        avg_grad_norm_v += gradient_norm_v

        do_norm_v = gradient_norm_v > self.norm_threshold and self.norm_grad
        kv = self.norm_threshold / gradient_norm_v if do_norm_v else 1.

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i] = (
                        self.lam_critic * self.z_critic[i] + (grads_v[i] * kv)
                )

        # --- Update Actor ETs --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        grads_pi = torch.autograd.grad(
            logp,
            list(self.actor.parameters()),
            retain_graph=False
        )

        flat_grads_pi = [g.flatten() for g in grads_pi]
        gradient_norm_pi = float(torch.norm(torch.cat(flat_grads_pi)))
        avg_grad_norm_pi += gradient_norm_pi

        do_norm_pi = gradient_norm_pi > self.norm_threshold and self.norm_grad
        kpi = self.norm_threshold / gradient_norm_pi if do_norm_pi else 1.

        with torch.no_grad():
            for i in range(len(self.z_actor)):
                self.z_actor[i] = (
                        self.lam_actor * self.z_actor[i] + (grads_pi[i] * kpi)
                )

        # ---- Update Critic --- #
        with torch.no_grad():
            for z, w in zip(self.z_critic, self.critic.parameters()):
                w += alpha_critic * delta * z

        # ---- Update Actor ---- #
        with torch.no_grad():
            for z, th in zip(self.z_actor, self.actor.parameters()):
                th += alpha_actor * delta * z


        if isinstance(self.temp, NoiseSchedule):
            self.temp.step()

        self.t += 1

        return dict(
            estimated_avg_reward=float(self.R_bar.detach().cpu()),
            td_error=to_native(delta),
            agent_loss = 0.5 * (avg_grad_norm_v + avg_grad_norm_pi)
        )


####################################################
############## Continuous Action ###################
####################################################


class ReinforceContinuousAction(ContinuousActionSoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            update_coefficient: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            normalize_reward: bool = False,
            seed: Optional[int] = None
    ):
        assert isinstance(action_size, int)
        assert 0 < action_size

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, NoiseSchedule)

        super().__init__(
            state_size=state_size,
            action_size=action_size,
            discount=discount,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient = update_coefficient
        self._writer: Optional[SummaryWriter] = None
        self.buffer = []
        self.discount = discount
        self.policy: Optional[GaussianPolicy] = None
        self.normalize_reward = normalize_reward
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold

    def init_model(self, *args, **kwargs):
        self.policy = GaussianPolicy(
            in_size=self.feature_size,
            action_size=self.action_size,
            hidden_dims=self.hidden_dims,
            normalize_input=self.normalize_input
        )

        self.optimizer = optim.SGD(
            self.policy.parameters(),
            lr=1e-3  # This will be overriden during learning
        )

    def initialize(self, **kwargs):

        if isinstance(self.update_coefficient, NoiseSchedule):
            self.update_coefficient.initialize()

        self.init_model()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.buffer = []

    def act(
            self,
            state,
            native = True
    ) -> Tuple[Union[Tuple, torch.Tensor], Union[Tuple, torch.Tensor]]:

        if isinstance(state, np.ndarray):
            pass
        elif isinstance(state, (list, tuple)):
            state = np.ndarray(state)
        elif isinstance(state, (int, float)):
            state = np.array([[state]])
        else:
            raise Exception("State datatype not recognized")

        if state.ndim == 1:
            state = state[None, ...]

        with torch.no_grad():
            actions, probs = self.policy.sample(
                to_tensor(state, dtype=torch.float32),
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)
            if isinstance(actions, (int, float)):
                actions = [actions]
                probs = [probs]

        return actions, probs

    def get_sa_probability(self, s, a, native = True):
        s, a = to_tensor_state_action(s, a)
        p = self.policy.prob_sa(s, a)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native = True):
        a, p = None, None

        with torch.no_grad():
            a, p = self.policy.greedy_action(to_tensor(s, dtype=torch.float32))

        if native:
            a = to_native(a)
            p = to_native(p)

            if isinstance(a, (int, float)):
                a = [a]
                p = [p]

        return a, p

    def pd(self, s):
        return self.policy.pd(s)

    def logp_sa(self, s, a, native=True):
        res = self.policy.logprob_sa(
            s=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.float32)
        )
        return to_native(res) if native else res

    def entropy(self, s, native = True):
        e = self.policy.entropy(to_tensor(s, torch.float32))
        return to_native(e) if native else e

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))
        if experience.done:
            self._learn()
            self.buffer.clear()
        self.t += 1

    def _learn(self):

        if isinstance(self.update_coefficient, NoiseSchedule):
            alpha = self.update_coefficient.value
            self.update_coefficient.step()
        else:
            alpha = self.update_coefficient

        T = len(self.buffer)
        Gs = np.zeros((T, ))

        R = [e.r for e, _ in self.buffer] if self.normalize_reward else 1.
        R, sig = (np.mean(R), np.std(R)) if self.normalize_reward else (0., 1.)

        def _G_update(i, r, done):
            nr = (r - R)/(sig + 1e-8)
            Gs[i] = nr if done else nr + self.discount * Gs[i+1]

        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        Gs = Gs - np.mean(Gs)

        total_grad_norm = 0.

        for experience, t in self.buffer:

            for g in self.optimizer.param_groups:
                g['lr'] = alpha

            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            logp = self.logp_sa(
                to_tensor(s, dtype=torch.float32),
                to_tensor(a, dtype=torch.float32),
                native=False
            )

            # Define the loss. We want to maximize log_prob * G_t,
            # so we minimize its negative.
            loss = -logp * (self.discount ** t) * Gs[t]
            self.optimizer.zero_grad()
            loss.backward()

            all_grads = torch.cat([
                p.grad.flatten()
                for p in self.policy.parameters()
                if p.grad is not None
            ])

            total_grad_norm += float(torch.norm(all_grads))

            # Clip the gradients to a max norm
            if self.norm_grad:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.norm_threshold
                )

            self.optimizer.step()  # PyTorch applies the update w += -lr * w.grad

        return total_grad_norm / len(self.buffer)


class ReinforceBaselineContinuousAction(
    ContinuousActionSoftPolicy,
    ContinuousActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            update_coefficient_policy: Union[float, NoiseSchedule],
            update_coefficient_baseline: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            norm_grad: bool = False,
            normalize_input: bool = False,
            norm_threshold: float = 10.,
            normalize_reward: bool = False,
            seed: Optional[int] = None
    ):
        assert isinstance(action_size, int)
        assert 0 < action_size

        if isinstance(update_coefficient_policy, float):
            assert 0. < update_coefficient_policy < 1.
        else:
            assert isinstance(update_coefficient_policy, NoiseSchedule)

        if isinstance(update_coefficient_baseline, float):
            assert 0. < update_coefficient_baseline < 1.
        else:
            assert isinstance(update_coefficient_baseline, NoiseSchedule)

        super().__init__(
            state_size=state_size,
            action_size=action_size,
            discount=discount,
            seed=seed
        )

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_policy
        self.update_coefficient_baseline = update_coefficient_baseline
        self._writer: Optional[SummaryWriter] = None
        self.buffer = []
        self.policy: Optional[GaussianPolicy] = None
        self.normalize_reward = normalize_reward
        self.norm_grad = norm_grad
        self.normalize_input = normalize_input
        self.norm_threshold = norm_threshold

    def init_model(self, *args, **kwargs):
        self.policy = GaussianPolicy(
            in_size=self.feature_size,
            action_size=self.action_size,
            hidden_dims=self.hidden_dims
        )

        # Using policy as feature extractor. One feature per action
        self.baseline = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

        self.policy_optimizer = optim.SGD(
            self.policy.parameters(),
            lr=1e-3
        )

        self.baseline_optimizer = optim.SGD(
            self.baseline.parameters(),
            lr=1e-3
        )

    def initialize(self, **kwargs):

        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_baseline, NoiseSchedule):
            self.update_coefficient_baseline.initialize()

        self.init_model()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.buffer = []

    def act(self, state, native = True) -> Tuple[Union[Tuple, torch.Tensor], Union[Tuple, torch.Tensor]]:

        if isinstance(state, np.ndarray):
            pass
        elif isinstance(state, (list, tuple)):
            state = np.ndarray(state)
        elif isinstance(state, (int, float)):
            state = np.array([[state]])
        else:
            raise Exception("State datatype not recognized")

        if state.ndim == 1:
            state = state[None, ...]

        with torch.no_grad():
            actions, probs = self.policy.sample(
                to_tensor(state, dtype=torch.float32),
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)
            if isinstance(actions, (int, float)):
                actions = [actions]
                probs = [probs]

        return actions, probs

    def get_sa_probability(self, s, a, native = True):
        s, a = to_tensor_state_action(s, a)
        p = self.policy.prob_sa(s, a)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native = True):
        a, p = None, None

        with torch.no_grad():
            a, p = self.policy.greedy_action(to_tensor(s, dtype=torch.float32))

        if native:
            a = to_native(a)
            p = to_native(p)

            if isinstance(a, (int, float)):
                a = [a]
                p = [p]

        return a, p

    def pd(self, s):
        return self.policy.pd(s)

    def logp_sa(self, s, a, native=True):
        res = self.policy.logprob_sa(
            s=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.float32)
        )
        return to_native(res) if native else res

    def entropy(self, s, native = True):
        e = self.policy.entropy(to_tensor(s, torch.float32))
        return to_native(e) if native else e

    def state_value(self, s, native = True, **kwargs) -> float:
        # V(s)
        v = self.baseline(
            to_tensor(s, dtype=torch.float32)
        )

        return to_native(v) if native else v.detach().cpu()

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))
        if experience.done:
            self._learn()
            self.buffer.clear()
        self.t += 1

    def _learn(self):
        if isinstance(self.update_coefficient_policy, NoiseSchedule):
            alpha_policy = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_policy = self.update_coefficient_policy

        if isinstance(self.update_coefficient_baseline, NoiseSchedule):
            alpha_baseline = self.update_coefficient_baseline.value
            self.update_coefficient_baseline.step()
        else:
            alpha_baseline = self.update_coefficient_baseline

        T = len(self.buffer)
        Gs = np.zeros((T, ))

        R = [e.r for e, _ in self.buffer] if self.normalize_reward else 1.
        R, sig = (np.mean(R), np.std(R)) if self.normalize_reward else (0., 1.)

        def _G_update(i, r, done):
            nr = (r - R) / (sig + 1e-8)
            Gs[i] = nr if done else nr + self.discount * Gs[i + 1]


        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        Gs = Gs - np.mean(Gs)

        for experience, t in self.buffer:
            for g in self.policy_optimizer.param_groups:
                g['lr'] = alpha_policy

            for g in self.baseline_optimizer.param_groups:
                g['lr'] = alpha_baseline

            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            with torch.no_grad():
                delta = Gs[t] - self.baseline(to_tensor(s, dtype=torch.float32))

            # grad_L2(G - \hat{v}) = -0.5 * delta * grad_\hat{v}
            # Minimize loss: w[t+1] = w[t] - \alpha * grad_L2
            # w[t+1] = w[t] + delta * grad_\hat{v}
            baseline_loss = torch.nn.MSELoss()(
                input=self.baseline(to_tensor(s, dtype=torch.float32)),
                target=to_tensor([Gs[t]], dtype=torch.float32)
            )

            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            logp = self.logp_sa(
                to_tensor(s, dtype=torch.float32),
                to_tensor(a, dtype=torch.float32),
                native=False
            )

            # Maximize: J(theta): \theta[t+1] = \theta[t] + \alpha * grad_J(theta)
            # grad_J(theta) = G_t * grad_log_prob
            #  \theta[t+1] = \theta[t] + \alpha *  G_t * grad_log_prob
            policy_loss = -logp * delta * (self.discount ** t)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.policy_optimizer.step()  # PyTorch applies the update w += -lr * w.grad


class ACWithEligibilityTracesContinuousAction(
    ContinuousActionSoftPolicy,
    ContinuousActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            lam_actor: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            norm_grad: bool = False,
            device: Optional[Union[str, torch.device]] = None
    ):
        assert isinstance(action_size, int)
        assert 0 < action_size

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        super().__init__(
            feature_size=state_size,
            action_size=action_size)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_actor = update_coefficient_actor
        self.update_coefficient_critic = update_coefficient_critic
        self._writer: Optional[SummaryWriter] = None
        self.discount = discount
        self.policy: Optional[GaussianPolicy] = None
        self.norm_grad = norm_grad
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self.I = 1
        self.device = device

        if device is not None:
            self.to_tensor = partial(to_tensor, device=device)
            self.to_tensor_state_action = partial(to_tensor_state_action, device=device)
        else:
            self.to_tensor = to_tensor
            self.to_tensor_state_action = to_tensor_state_action

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    def init_model(self, *args, **kwargs):
        self.actor = GaussianPolicy(
            in_size=self.feature_size,
            action_size=self.action_size,
            hidden_dims=self.hidden_dims
        )

        self.actor = torch.compile(self.actor)

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

        self.critic = torch.compile(self.critic)

        if self.device is not None:
            self.actor.to(self.device)
            self.critic.to(self.device)

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

        self.actor_optimizer = optim.SGD(
            self.actor.parameters(),
            lr=1e-3
        )

        self.critic_optimizer = optim.SGD(
            self.critic.parameters(),
            lr=1e-3
        )

    def initialize(self, **kwargs):
        if isinstance(self.update_coefficient_actor, NoiseSchedule):
            self.update_coefficient_actor.initialize()

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            self.update_coefficient_critic.initialize()

        self.init_model()

        self.I = 1
        self.t = 0

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.I = 1

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def act(self, state, native = True) -> Tuple[Union[Tuple, torch.Tensor], Union[Tuple, torch.Tensor]]:

        if isinstance(state, np.ndarray):
            pass
        elif isinstance(state, (list, tuple)):
            state = np.ndarray(state)
        elif isinstance(state, (int, float)):
            state = np.array([[state]])
        else:
            raise Exception("State datatype not recognized")

        if state.ndim == 1:
            state = state[None, ...]

        with torch.no_grad():
            actions, probs = self.actor.sample(
                self.to_tensor(state, dtype=torch.float32),
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)
            if isinstance(actions, (int, float)):
                actions = [actions]
                probs = [probs]

        return actions, probs

    def get_sa_probability(self, s, a, native = True):
        s, a = self.to_tensor_state_action(s, a)
        p = self.actor.prob_sa(s, a)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native = True):
        a, p = None, None

        with torch.no_grad():
            a, p = self.actor.greedy_action(self.to_tensor(s, dtype=torch.float32))

        if native:
            a = to_native(a)
            p = to_native(p)

            if isinstance(a, (int, float)):
                a = [a]
                p = [p]

        return a, p

    def pd(self, s):
        return self.actor.pd(s)

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            s=self.to_tensor(s, dtype=torch.float32),
            a=self.to_tensor(a, dtype=torch.float32)
        )
        return to_native(res) if native else res

    def entropy(self, s, native = True):
        e = self.actor.entropy(self.to_tensor(s, torch.float32))
        return to_native(e) if native else e

    def step(self, experience: Experience, **kwargs):

        def _get_alpha(u):
            if isinstance(u, NoiseSchedule):
                a = u.value
                u.step()
            else:
                a = u
            return a

        alpha_actor = _get_alpha(self.update_coefficient_actor)
        alpha_critic = _get_alpha(self.update_coefficient_critic)

        s, a, r, sp, ap, done = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap, experience.done
        )

        v = self.critic(self.to_tensor(s))

        with torch.no_grad():
            vp = self.critic(self.to_tensor(sp)) * (1 - done)
            delta = (r + self.discount * vp) - v

        # --- Update Critic ETs --- #
        grads_v = torch.autograd.grad(
            v,
            list(self.critic.parameters()),
            retain_graph=False
        )

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i].data.copy_(
                        self.discount * self.lam_critic * self.z_critic[i] +
                        grads_v[i]
                )

        # --- Update Actor ETs --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        grads_pi = torch.autograd.grad(
            logp,
            list(self.actor.parameters()),
            retain_graph=False
        )

        with torch.no_grad():
            for i in range(len(self.z_actor)):
                self.z_actor[i].data.copy_(
                        self.discount * self.lam_actor * self.z_actor[i] +
                        self.I * grads_pi[i]
                )

        # ---- Update Critic --- #
        for g in self.critic_optimizer.param_groups:
            g['lr'] = alpha_critic

        # self.critic_optimizer.zero_grad()
        for z, w in zip(self.z_critic, self.critic.parameters()):
            if w.grad is None:
                w.grad = torch.zeros_like(z)
            w.grad.data.copy_(-delta * z) # -grad because optimizer subtracts
        self.critic_optimizer.step()

        # ---- Update Actor ---- #
        for g in self.actor_optimizer.param_groups:
            g['lr'] = alpha_actor

        # self.actor_optimizer.zero_grad()
        for z, th in zip(self.z_actor, self.actor.parameters()):
            if th.grad is None:
                th.grad = torch.zeros_like(z)
            th.grad.data.copy_(-delta * z) # -grad because optimizer subtracts
        self.actor_optimizer.step()

        self.I *= self.discount
        self.t += 1


class ACWithEligibilityTracesContinuousActionContinuingTask(
    ContinuousActionSoftPolicy,
    ContinuousActionCriticStateValue
):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            update_coefficient_actor: Union[float, NoiseSchedule],
            lam_actor: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            update_coefficient_avg_reward: Union[float, NoiseSchedule],
            hidden_dims=(32,),
            norm_grad: bool = False,
            device: Optional[Union[str, torch.device]] = None
    ):
        assert isinstance(action_size, int)
        assert 0 < action_size

        if isinstance(update_coefficient_actor, float):
            assert 0. < update_coefficient_actor < 1.
        else:
            assert isinstance(update_coefficient_actor, NoiseSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, NoiseSchedule)

        if isinstance(update_coefficient_avg_reward, float):
            assert 0. < update_coefficient_avg_reward < 1.
        else:
            assert isinstance(update_coefficient_avg_reward, NoiseSchedule)

        super().__init__(
            feature_size=state_size,
            action_size=action_size)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_actor = update_coefficient_actor
        self.update_coefficient_critic = update_coefficient_critic
        self.update_coefficient_avg_reward = update_coefficient_avg_reward
        self._writer: Optional[SummaryWriter] = None
        self.policy: Optional[GaussianPolicy] = None
        self.norm_grad = norm_grad
        self.lam_actor = lam_actor
        self.lam_critic = lam_critic
        self.device = device
        self.R_bar = 0

        if device is not None:
            self.to_tensor = partial(to_tensor, device=device)
            self.to_tensor_state_action = partial(
                to_tensor_state_action, device=device)
        else:
            self.to_tensor = to_tensor
            self.to_tensor_state_action = to_tensor_state_action

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    def init_model(self, *args, **kwargs):
        self.actor = GaussianPolicy(
            in_size=self.feature_size,
            action_size=self.action_size,
            hidden_dims=self.hidden_dims
        )

        self.actor = torch.compile(self.actor)

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

        self.critic = torch.compile(self.critic)

        if self.device is not None:
            self.actor.to(self.device)
            self.critic.to(self.device)

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

        self.actor_optimizer = optim.SGD(
            self.actor.parameters(),
            lr=1e-3
        )

        self.critic_optimizer = optim.SGD(
            self.critic.parameters(),
            lr=1e-3
        )

    def initialize(self, **kwargs):
        if isinstance(self.update_coefficient_actor, NoiseSchedule):
            self.update_coefficient_actor.initialize()

        if isinstance(self.update_coefficient_critic, NoiseSchedule):
            self.update_coefficient_critic.initialize()

        self.init_model()

        self.t = 0
        self.R_bar = 0

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.R_bar = 0

        self.z_actor = [
            torch.zeros_like(p).to(p.device)
            for p in self.actor.parameters()
        ]
        self.z_critic = [
            torch.zeros_like(p).to(p.device)
            for p in self.critic.parameters()
        ]

    def act(self, state, native=True) -> Tuple[Union[Tuple, torch.Tensor], Union[Tuple, torch.Tensor]]:

        if isinstance(state, np.ndarray):
            pass
        elif isinstance(state, (list, tuple)):
            state = np.ndarray(state)
        elif isinstance(state, (int, float)):
            state = np.array([[state]])
        else:
            raise Exception("State datatype not recognized")

        if state.ndim == 1:
            state = state[None, ...]

        with torch.no_grad():
            actions, probs = self.actor.sample(
                self.to_tensor(state, dtype=torch.float32),
                differentiable=False)

        if native:
            actions, probs = to_native(actions), to_native(probs)
            if isinstance(actions, (int, float)):
                actions = [actions]
                probs = [probs]

        return actions, probs

    def get_sa_probability(self, s, a, native=True):
        s, a = self.to_tensor_state_action(s, a)
        p = self.actor.prob_sa(s, a)
        return to_native(p) if native else p

    def get_greedy_action(self, s, native=True):
        a, p = None, None

        with torch.no_grad():
            a, p = self.actor.greedy_action(
                self.to_tensor(s, dtype=torch.float32))

        if native:
            a = to_native(a)
            p = to_native(p)

            if isinstance(a, (int, float)):
                a = [a]
                p = [p]

        return a, p

    def pd(self, s):
        return self.actor.pd(s)

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            s=self.to_tensor(s, dtype=torch.float32),
            a=self.to_tensor(a, dtype=torch.float32)
        )
        return to_native(res) if native else res

    def entropy(self, s, native=True):
        e = self.actor.entropy(self.to_tensor(s, torch.float32))
        return to_native(e) if native else e

    def step(self, experience: Experience, **kwargs):

        def _get_alpha(u):
            if isinstance(u, NoiseSchedule):
                a = u.value
                u.step()
            else:
                a = u
            return a

        alpha_actor = _get_alpha(self.update_coefficient_actor)
        alpha_critic = _get_alpha(self.update_coefficient_critic)
        alpha_r = _get_alpha(self.update_coefficient_avg_reward)

        s, a, r, sp, ap = (
            experience.s, experience.a,
            experience.r, experience.sp,
            experience.ap
        )

        v = self.critic(self.to_tensor(s))

        with torch.no_grad():
            vp = self.critic(self.to_tensor(sp))
            delta = (r - self.R_bar * vp) - v


        self.R_bar += alpha_r * float(delta)

        # --- Update Critic ETs --- #
        grads_v = torch.autograd.grad(
            v,
            list(self.critic.parameters()),
            retain_graph=False
        )

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i].data.copy_(
                        self.lam_critic * self.z_critic[i] + grads_v[i]
                )

        # --- Update Actor ETs --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        grads_pi = torch.autograd.grad(
            logp,
            list(self.actor.parameters()),
            retain_graph=False
        )

        with torch.no_grad():
            for i in range(len(self.z_actor)):
                self.z_actor[i].data.copy_(
                        self.lam_actor * self.z_actor[i] + grads_pi[i]
                )

        # ---- Update Critic --- #
        for g in self.critic_optimizer.param_groups:
            g['lr'] = alpha_critic

        # self.critic_optimizer.zero_grad()
        for z, w in zip(self.z_critic, self.critic.parameters()):
            if w.grad is None:
                w.grad = torch.zeros_like(z)
            w.grad.data.copy_(-delta * z) # -grad because optimizer subtracts
        self.critic_optimizer.step()

        # ---- Update Actor ---- #
        for g in self.actor_optimizer.param_groups:
            g['lr'] = alpha_actor

        # self.actor_optimizer.zero_grad()
        for z, th in zip(self.z_actor, self.actor.parameters()):
            if th.grad is None:
                th.grad = torch.zeros_like(z)
            th.grad.data.copy_(-delta * z) # -grad because optimizer subtracts
        self.actor_optimizer.step()

        self.t += 1

