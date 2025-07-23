from typing import Union, Callable, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from approximate_methods.utils import (
    DiscreteActionAgent,
    NoiseSchedule,
    Experience)
from shared.utils import LinearSchedule, SoftPolicy

from policy_gradient.nets import DiscreteActionPolicyMLP, ValueFunction

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


class Reinforce_LA(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            feature_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            policy_feature_fn: Callable[[Any, ], np.ndarray], # state --> np.ndarray
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, LinearSchedule] = 1.,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

        super().__init__(
            feature_size=feature_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.update_coefficient = update_coefficient
        self._writer: Optional[SummaryWriter] = None
        self.buffer = []
        self.discount = discount
        self.temp = temp
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

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.initialize()

        self.init_weights()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        if isinstance(self.temp, NoiseSchedule):
            self.temp.reset()

        if isinstance(self.update_coefficient, LinearSchedule):
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

        a = np.random.choice(acts, replace=True, p=probs)

        if isinstance(self.temp, LinearSchedule):
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
            return int(np.random.choice(idc)), 1. / len(idc)
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

        if isinstance(self.update_coefficient, LinearSchedule):
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

        if isinstance(self.temp, LinearSchedule):
            self.temp.step()

# ======================================================================== #

def to_tensor(s, dtype=None):
    if not isinstance(s, np.ndarray):
        assert dtype is not None
        s = torch.tensor(s, dtype=dtype)
    else:
        s = torch.from_numpy(s)
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


def to_tensor_state_action(s, a):
    if isinstance(a, np.ndarray):
        assert a.dtype in (np.int32, np.int64)
        assert isinstance(s, np.ndarray)
        assert a.shape[0] == s.shape[0]
        s = to_tensor(s)
    elif isinstance(a, int):
        a = to_tensor([a], torch.long)
        assert not isinstance(s, np.ndarray)
        s = to_tensor(s, dtype=torch.float32)
    elif isinstance(a, torch.Tensor):
        assert isinstance(s, torch.Tensor)
    else:
        raise Exception("Input types not recognized")
    
    return s, a


class Reinforce(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, LinearSchedule] = 1.,
            norm_grad: bool = False,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient, float):
            assert 0. < update_coefficient < 1.
        else:
            assert isinstance(update_coefficient, LinearSchedule)

        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient = update_coefficient
        self._writer: Optional[SummaryWriter] = None
        self.buffer = []
        self.discount = discount
        self.temp = temp
        self.policy = None
        self.norm_grad = norm_grad

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    def init_model(self, *args, **kwargs):
        self.policy = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient, LinearSchedule):
            self.update_coefficient.initialize()

        self.init_model()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        # if isinstance(self.temp, NoiseSchedule):
        #     self.temp.reset()

        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()

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

        for experience, t in self.buffer:
            if isinstance(self.update_coefficient, LinearSchedule):
                alpha = self.update_coefficient.value
                self.update_coefficient.step()
            else:
                alpha = self.update_coefficient


            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            logp = self.logp_sa(s, [a], native=False)
            assert not torch.any(torch.isnan(logp))

            weights = list(self.policy.parameters())
            grads = torch.autograd.grad(logp, weights, retain_graph=False)

            with torch.no_grad():
                for w, g in zip(weights, grads):
                    update = alpha * (self.discount ** t) * Gs[t]
                    update *= (
                        F.normalize(g, p=2, dim=-1)
                        if self.norm_grad else g
                    )
                    w += update

        if isinstance(self.temp, LinearSchedule):
            self.temp.step()


class ReinforceBaseline(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_policy: Union[float, NoiseSchedule],
            update_coefficient_baseline: Union[float, NoiseSchedule],
            hidden_dims=(32, ),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, LinearSchedule] = 1.,
            norm_grad: bool = False,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_policy, float):
            assert 0. < update_coefficient_policy < 1.
        else:
            assert isinstance(update_coefficient_policy, LinearSchedule)


        if isinstance(update_coefficient_baseline, float):
            assert 0. < update_coefficient_baseline < 1.
        else:
            assert isinstance(update_coefficient_baseline, LinearSchedule)

        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_policy
        self.update_coefficient_baseline = update_coefficient_baseline
        self._writer: Optional[SummaryWriter] = None
        self.buffer = []
        self.discount = discount
        self.temp = temp
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    def init_model(self, *args, **kwargs):
        self.policy = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims
        )

        # Using policy as feature extractor. One feature per action
        self.value = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_baseline, LinearSchedule):
            self.update_coefficient_baseline.initialize()

        self.init_model()

        self.buffer = []

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0

        # if isinstance(self.temp, NoiseSchedule):
        #     self.temp.reset()

        # if isinstance(self.update_coefficient, LinearSchedule):
        #     self.update_coefficient.reset()

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

    def step(self, experience: Experience, **kwargs):
        self.buffer.append((experience, self.t))
        if experience.done:
            self._learn()
            self.buffer.clear()
        self.t += 1

    def _learn(self):
        T = len(self.buffer)
        Gs = np.zeros((T,))

        def _G_update(i, r, done):
            Gs[i] = r if done else r + self.discount * Gs[i + 1]

        # Generate G_t's
        _ = [
            _G_update(_t, e.r, e.done)
            for e, _t in reversed(self.buffer)
        ]

        for experience, t in self.buffer:
            if isinstance(self.update_coefficient_policy, LinearSchedule):
                alpha_pi = self.update_coefficient_policy.value
                self.update_coefficient_policy.step()
            else:
                alpha_pi = self.update_coefficient_policy

            if isinstance(self.update_coefficient_baseline, LinearSchedule):
                alpha_b = self.update_coefficient_baseline.value
                self.update_coefficient_baseline.step()
            else:
                alpha_b = self.update_coefficient_baseline

            s, a, r, sp, ap, done = (
                experience.s, experience.a,
                experience.r, experience.sp,
                experience.ap, experience.done
            )

            # --- Update baseline --- #
            v = self.value(to_tensor(s, dtype=torch.float32))
            delta = Gs[t] - v
            W = list(self.value.parameters())
            grads_v = torch.autograd.grad(v, W, retain_graph=False)

            with torch.no_grad():
                for w, g in zip(W, grads_v):
                    update = alpha_b * delta * (
                        F.normalize(g, p=2, dim=-1)
                        if self.norm_grad else g
                    )
                    w += update

            # --- Update Actor --- #
            logp = self.logp_sa(s, [a], native=False)
            assert not torch.any(torch.isnan(logp))
            theta = list(self.policy.parameters())
            grads_pi = torch.autograd.grad(logp, theta, retain_graph=False)

            with torch.no_grad():
                for th, g in zip(theta, grads_pi):
                    update = alpha_pi * (self.discount ** t) * delta * (
                        F.normalize(g, p=2, dim=-1)
                        if self.norm_grad else g
                    )
                    th += update

        if isinstance(self.temp, LinearSchedule):
            self.temp.step()


class OneStepAC(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_policy: Union[float, NoiseSchedule],
            update_coefficient_critic: Union[float, NoiseSchedule],
            hidden_dims=(32,),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, LinearSchedule] = 1.,
            norm_grad: bool = False,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_policy, float):
            assert 0. < update_coefficient_policy < 1.
        else:
            assert isinstance(update_coefficient_policy, LinearSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, LinearSchedule)

        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_policy
        self.update_coefficient_critic = update_coefficient_critic
        self._writer: Optional[SummaryWriter] = None
        self.discount = discount
        self.temp = temp
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.I = 1

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, LinearSchedule):
            self.update_coefficient_critic.initialize()

        self.init_model()
        self.I = 1
        self.t = 0

    def reset(self):
        # The agent here is prepared for a new episode
        self.t = 0
        self.I = 1

    def logp_sa(self, s, a, native=True):
        res = self.actor.logprob_sa(
            x=to_tensor(s, dtype=torch.float32),
            a=to_tensor(a, dtype=torch.long),
            temperature=self.temperature
        )
        return to_native(res) if native else res

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

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, LinearSchedule):
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

        # --- Update critic --- #
        W = list(self.critic.parameters())
        grads_v = torch.autograd.grad(v, W, retain_graph=False)

        with torch.no_grad():
            for w, g in zip(W, grads_v):
                update = alpha_critic * delta * (
                    F.normalize(g, p=2, dim=-1)
                    if self.norm_grad else g
                )
                w += update


        # --- Update Actor --- #
        logp = self.logp_sa(s, [a], native=False)
        assert not torch.any(torch.isnan(logp))
        theta = list(self.actor.parameters())
        grads_pi = torch.autograd.grad(logp, theta, retain_graph=False)

        with torch.no_grad():
            for th, g in zip(theta, grads_pi):
                update = alpha_actor * self.I * delta * (
                    F.normalize(g, p=2, dim=-1)
                    if self.norm_grad else g
                )
                th += update

        if isinstance(self.temp, LinearSchedule):
            self.temp.step()

        self.I *= self.discount
        self.t += 1


class ACWithEligibilityTraces(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_policy: Union[float, NoiseSchedule],
            lam_policy: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            hidden_dims=(32,),
            discount: Union[float, NoiseSchedule] = 0.9,
            temp: Union[float, LinearSchedule] = 1.,
            norm_grad: bool = False,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_policy, float):
            assert 0. < update_coefficient_policy < 1.
        else:
            assert isinstance(update_coefficient_policy, LinearSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, LinearSchedule)

        assert 0 <= lam_policy <= 1
        assert 0 <= lam_critic <= 1

        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_policy
        self.update_coefficient_critic = update_coefficient_critic
        self.lam_actor = lam_policy
        self.lam_critic = lam_critic
        self._writer: Optional[SummaryWriter] = None
        self.discount = discount
        self.temp = temp
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.I = 1
        self.z_critic = None
        self.z_actor = None

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, LinearSchedule):
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

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, LinearSchedule):
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

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i] = (
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
                self.z_actor[i] = (
                        self.discount * self.lam_actor * self.z_actor[i] +
                        self.I * grads_pi[i]
                )

        # ---- Update Critic --- #
        with torch.no_grad():
            for z, w in zip(self.z_critic, self.critic.parameters()):
                w += alpha_critic * delta * z

        # ---- Update Actor ---- #
        with torch.no_grad():
            for z, th in zip(self.z_actor, self.actor.parameters()):
                th += alpha_actor * delta * z


        if isinstance(self.temp, LinearSchedule):
            self.temp.step()

        self.I *= self.discount
        self.t += 1


class ACWithEligibilityTracesContinuing(DiscreteActionAgent, SoftPolicy):
    def __init__(
            self,
            state_size: int,
            action_space_dims: int,
            update_coefficient_policy: Union[float, NoiseSchedule],
            lam_policy: float,
            update_coefficient_critic: Union[float, NoiseSchedule],
            lam_critic: float,
            update_coefficient_avg_reward: Union[float, NoiseSchedule],
            hidden_dims=(32,),
            temp: Union[float, LinearSchedule] = 1.,
            norm_grad: bool = False,
    ):
        assert 0 < action_space_dims
        assert isinstance(action_space_dims, int)

        if isinstance(update_coefficient_policy, float):
            assert 0. < update_coefficient_policy < 1.
        else:
            assert isinstance(update_coefficient_policy, LinearSchedule)

        if isinstance(update_coefficient_critic, float):
            assert 0. < update_coefficient_critic < 1.
        else:
            assert isinstance(update_coefficient_critic, LinearSchedule)

        if isinstance(update_coefficient_avg_reward, float):
            assert 0. < update_coefficient_avg_reward < 1.
        else:
            assert isinstance(update_coefficient_avg_reward, LinearSchedule)

        assert 0 <= lam_policy <= 1
        assert 0 <= lam_critic <= 1

        super().__init__(
            feature_size=state_size,
            action_space_dims=action_space_dims)

        self.t = 0
        self.hidden_dims = hidden_dims
        self.update_coefficient_policy = update_coefficient_policy
        self.update_coefficient_critic = update_coefficient_critic
        self.update_coefficient_avg_reward = update_coefficient_avg_reward
        self.lam_actor = lam_policy
        self.lam_critic = lam_critic
        self._writer: Optional[SummaryWriter] = None
        self.temp = temp
        self.policy = None
        self.value = None
        self.norm_grad = norm_grad
        self.z_critic = None
        self.z_actor = None
        self.R_bar = 0

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @writer.setter
    def writer(self, w: SummaryWriter):
        if w is not None:
            assert isinstance(w, SummaryWriter)
        self._writer = w

    @property
    def temperature(self) -> float:
        t = self.temp
        if isinstance(t, LinearSchedule):
            t = t.value
        return t

    def init_model(self, *args, **kwargs):
        self.actor = DiscreteActionPolicyMLP(
            in_size=self.feature_size,
            n_actions=self.action_space_dims,
            hidden_dims=self.hidden_dims
        )

        # Using policy as feature extractor. One feature per action
        self.critic = ValueFunction(
            in_size=self.feature_size,
            hidden_dims=self.hidden_dims
        )

    def initialize(self, **kwargs):
        if isinstance(self.temp, NoiseSchedule):
            # Reset noise to starting exploration
            self.temp.initialize()

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            self.update_coefficient_policy.initialize()

        if isinstance(self.update_coefficient_critic, LinearSchedule):
            self.update_coefficient_critic.initialize()

        if isinstance(self.update_coefficient_avg_reward, LinearSchedule):
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

        if isinstance(self.update_coefficient_policy, LinearSchedule):
            alpha_actor = self.update_coefficient_policy.value
            self.update_coefficient_policy.step()
        else:
            alpha_actor = self.update_coefficient_policy

        if isinstance(self.update_coefficient_critic, LinearSchedule):
            alpha_critic = self.update_coefficient_critic.value
            self.update_coefficient_critic.step()
        else:
            alpha_critic = self.update_coefficient_critic

        if isinstance(self.update_coefficient_avg_reward, LinearSchedule):
            alpha_reward = self.update_coefficient_avg_reward.value
            self.update_coefficient_avg_reward.step()
        else:
            alpha_reward = self.update_coefficient_avg_reward

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

        with torch.no_grad():
            for i in range(len(self.z_critic)):
                self.z_critic[i] = (
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

        # print(f'Pi grad: {float(torch.concat([g.reshape(-1) for g in grads_pi]).norm()) :.2e} \n')
        # print(f'V grad: {float(torch.concat([g.reshape(-1) for g in grads_v]).norm()) :.2e} \n')
        # print(f'R_bar: {float(self.R_bar): .2e}\n')

        with torch.no_grad():
            for i in range(len(self.z_actor)):
                self.z_actor[i] = (
                        self.lam_actor * self.z_actor[i] + grads_pi[i]
                )

        # ---- Update Critic --- #
        with torch.no_grad():
            for z, w in zip(self.z_critic, self.critic.parameters()):
                w += alpha_critic * delta * z

        # ---- Update Actor ---- #
        with torch.no_grad():
            for z, th in zip(self.z_actor, self.actor.parameters()):
                th += alpha_actor * delta * z


        if isinstance(self.temp, LinearSchedule):
            self.temp.step()

        self.t += 1


