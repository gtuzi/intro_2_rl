from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import (
    Categorical,
    Normal,
    TanhTransform,
    AffineTransform,
    ComposeTransform,
    TransformedDistribution
)


Tensor = torch.Tensor

def init_weights(module):
    """
    Weight initializer helper for PyTorch layers.

    - Linear layers: Xavier uniform (Glorot) for weights, zeros for biases.
    - Conv2d layers: Kaiming normal for weights, zeros for biases.
    - BatchNorm layers: Ones for weights, zeros for biases.
    """
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        init.ones_(module.weight)
        init.zeros_(module.bias)


def sample_softmax(
        logits: torch.Tensor,
        temperature: float = 1.0,
        differentiable: bool = False,
        hard: bool = False
):
    """
    Sample from a softmax distribution given logits, returning both the sample and its probability.

    Parameters
    ----------
    logits : torch.Tensor, shape (..., n_classes)
        Unnormalized log‐probabilities.
    temperature : float
        Temperature for (Gumbel‐)Softmax. Only used if differentiable=True.
    differentiable : bool
        If False: draws a non‐differentiable categorical sample.
        If True: uses Gumbel‐Softmax to produce a differentiable sample.
    hard : bool
        Only used if differentiable=True. If True, performs a straight‐through one‐hot sample;
        if False, returns a soft probability vector.

    Returns
    -------
    sample : torch.Tensor
        If differentiable=False: shape (...,) of integer class indices.
        If differentiable=True: shape (..., n_classes) of (soft or hard) sample vectors.
    sample_prob : torch.Tensor
        The probability of each drawn sample (shape (...) matching sample indices or sample vectors).
    """

    # Compute the base softmax probabilities
    probs = F.softmax(logits/temperature, dim=-1)

    if not differentiable:
        # Non-differentiable categorical draw
        dist = Categorical(probs=probs)
        idx = dist.sample()  # shape (...)
        p_of_sample = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return idx, p_of_sample

    # Differentiable Gumbel-Softmax draw
    y = F.gumbel_softmax(
        logits,
        tau=temperature,
        hard=hard,
        dim=-1)  # shape (..., n_classes)

    # Probability of the drawn class:
    if hard:
        # For hard, y is one-hot, so dot with probs
        p_of_sample = (probs * y).sum(dim=-1)
    else:
        # For soft, y itself is the relaxed probability vector
        p_of_sample = y.sum(dim=-1) * 1.0  # essentially 1.0 per row, but kept for consistency
    return y, p_of_sample


class Backbone(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            hidden_dims: Optional[Union[int, List, Tuple]] = None,
            normalize_input: bool = False
    ):
        super(Backbone, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_dims = hidden_dims
        self.normalize_input = normalize_input
        self.net = None
        self._build_net()

    def _build_net(self):
        if self.hidden_dims is not None:
            layers = []

            if self.normalize_input:
                layers += [nn.LayerNorm(self.in_size)]

            # Add input layer
            layers += [
                nn.Linear(
                    in_features=self.in_size,
                    out_features=self.hidden_dims[0]
                ),
                nn.ReLU()
            ]

            # Add hidden layers
            for i, _ in enumerate(self.hidden_dims[1:]):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                layers.append(nn.ReLU())

            # Add output layer
            layers.append(nn.Linear(
                in_features=self.hidden_dims[-1],
                out_features=self.out_size)
            )

        else:
            layers = [
                nn.Linear(self.in_size, self.out_size),
            ]

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x: Tensor):
        return self.net(x)


class DiscreteActionPolicyMLP(nn.Module):
    def __init__(
            self,
            in_size: int,
            n_actions: int,
            hidden_dims: Optional[Union[int, List, Tuple]] = None,
            normalize_input: bool = False
    ):
        super(DiscreteActionPolicyMLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.in_size = in_size
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.normalize_input = normalize_input
        self.net = None
        self._build_net()

    def _build_net(self, backbone = None):
        if backbone is None:
            self.net = Backbone(
                in_size=self.in_size,
                hidden_dims=self.hidden_dims,
                out_size=self.n_actions,
                normalize_input=self.normalize_input
            )
        else:
            assert backbone.out_size == self.n_actions
            self.net = backbone

    def logprob_s(self, x: Tensor, temperature: float = 1.):
        return F.log_softmax(self.forward(x)/temperature, dim=-1)

    def prob_s(self, x: Tensor, temperature: float = 1.):
        return F.softmax(self.forward(x) / temperature, dim=-1)

    def prob_sa(self, x: Tensor, a: Tensor, temperature: float = 1.):
        """
            a values in [0, num_actions)
        """
        ps = self.prob_s(x, temperature)
        if a.ndim == 1:
            a = a.unsqueeze(1)
        assert a.ndim == 2
        return ps.gather(dim=1, index=a).squeeze(1)

    def logprob_sa(self, x: Tensor, a: Tensor, temperature: float = 1.):
        lps = self.logprob_s(x, temperature=temperature)
        if a.ndim < lps.ndim:
            a = a.unsqueeze(1)
            assert a.ndim == 2
        res = lps.gather(dim=-1, index=a)
        return res

    def entropy(self, x: Tensor, temperature: float = 1.):
        logp = self.logprob_s(x, temperature)
        p = torch.exp(logp)
        return -(p * logp).sum(dim=-1)

    def sample(self, x: Tensor, temperature: float = 1., differentiable=False):
        s, p = sample_softmax(
            logits=self.forward(x),
            temperature=temperature,
            differentiable=differentiable
        )

        return s, p

    def forward(self, x: Tensor):
        return self.net(x)


class ValueFunction(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_dims: Optional[Union[int, List, Tuple]] = None,
            normalize_input: bool = False
    ):
        super(ValueFunction, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.in_size = in_size
        self.hidden_dims = hidden_dims
        self.normalize_input = normalize_input
        self.net = None
        self._build_net()

    def _build_net(self):
        self.net = Backbone(
            in_size=self.in_size,
            hidden_dims=self.hidden_dims,
            out_size=1,
            normalize_input=self.normalize_input
        )

    def forward(self, x: Tensor):
        return self.net(x)


class ContinuousActionPolicy(nn.Module):
    def __init__(
            self,
            in_size: int,
            action_size: int,
            hidden_dims: Optional[Union[int, List, Tuple]] = None,
            normalize_input: bool = False
    ):
        super(ContinuousActionPolicy, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.in_size = in_size
        self.action_size = action_size
        self.hidden_dims = hidden_dims
        self.normalize_input = normalize_input
        self.net = None
        self.loc = None
        self.scale = None
        self._build_net()

    def _build_net(self, backbone=None):
        if backbone is None:

            self.loc = Backbone(
                in_size=self.in_size,
                hidden_dims=self.hidden_dims,
                out_size=self.action_size,
                normalize_input=self.normalize_input
            )

            self.scale = Backbone(
                in_size=self.in_size,
                hidden_dims=self.hidden_dims,
                out_size=self.action_size,
                normalize_input=self.normalize_input
            )
        else:
            raise NotImplementedError

    def pd(self, s: Tensor):
        raise NotImplementedError

    def greedy_action(self, s: Tensor):
        raise NotImplementedError

    def mean(self, s: Tensor):
        raise NotImplementedError

    def sigma(self, s: Tensor):
        raise NotImplementedError

    def prob_sa(self, s: Tensor, a: Tensor):
        return torch.exp(self.pd(s).log_prob(a))

    def logprob_sa(self, s: Tensor, a: Tensor):
        return self.pd(s).log_prob(a)

    def entropy(self, s: Tensor):
        raise NotImplementedError

    def sample(self, s: Tensor, differentiable=False):
        pd = self.pd(s)
        if differentiable:
            sample = pd.rsample()
        else:
            sample = pd.sample()

        return sample, torch.exp(pd.log_prob(sample))

    def forward(self, x: Tensor):
        raise NotImplementedError


class GaussianPolicy(ContinuousActionPolicy):

    def __init__(
            self,
            in_size: int,
            action_size: int,
            action_mins: Tuple[float, ...] = None,
            action_maxs: Tuple[float, ...] = None,
            hidden_dims: Optional[Union[int, List, Tuple]] = None,
            normalize_input: bool = False
    ):
        super(GaussianPolicy, self).__init__(
            in_size=in_size,
            action_size=action_size,
            hidden_dims=hidden_dims,
            normalize_input=normalize_input
        )

        self.action_mins = action_mins
        self.action_maxs = action_maxs

        if (self.action_mins is not None) or (self.action_maxs is not None):
            assert self.action_mins is not None
            assert self.action_maxs is not None
            assert len(self.action_mins) == len(self.action_maxs)

            scales = [
                (mx - mi) / 2
                for mi, mx in zip(self.action_mins, self.action_maxs)
            ]

            biases = [
                (mx + mi) / 2
                for mi, mx in zip(self.action_mins, self.action_maxs)
            ]

            # Calculate the scale and shift needed to
            # map [-1, 1] to [low, high]
            self.action_scale = torch.tensor(scales, dtype=torch.float32)
            self.action_bias = torch.tensor(biases, dtype=torch.float32)


    def pd(self, s: Tensor):
        res = self.forward(s)
        mu, sig = res[..., :self.action_size], res[..., self.action_size:]
        pd = Normal(mu, sig)

        if (self.action_mins is not None) and (self.action_maxs is not None):

            transforms = [
                # First, squash the output to the [-1, 1] range
                TanhTransform(cache_size=1),
                # Second, apply scaling and shifting to match the environment's action space
                AffineTransform(
                    loc=self.action_bias,
                    scale=self.action_scale,
                    cache_size=1)
            ]

            return TransformedDistribution(pd, transforms)
        else:
            return pd

    def entropy(self, s: Tensor):
        """Returns the entropy of the base (pre-squashed) distribution."""
        res = self.forward(s)
        mu, sig = res[..., :self.action_size], res[..., self.action_size:]
        pd = Normal(mu, sig)
        return pd.entropy()

    def greedy_action(self, s: Tensor):
        return self.mean(s)

    def mean(self, s: Tensor):
        res = self.forward(s)
        mu, sig = res[..., :self.action_size], res[..., self.action_size:]
        pd = Normal(mu, sig)

        if (self.action_mins is not None) and (self.action_maxs is not None):

            # Transform and shift accordingly
            greedy_action = torch.tanh(
                mu) * self.action_scale + self.action_bias

            # The probability of the greedy action is not a well-defined
            return greedy_action, None
        else:
            return mu, torch.exp(pd.log_prob(mu))

    def sigma(self, s: Tensor):
        res = self.forward(s)
        # Return the sigma of the base distribution
        # regardless of transformations
        return res[..., self.action_size:]

    def forward(self, x: Tensor):
        mu = self.loc(x)
        # Technically exp() is needed here. But it's unstable to learn with
        # as it can swing pretty wildly, if the outputs of self.scale vary
        # a lot, causing wild swings accross gradients.
        sig = F.softplus(self.scale(x)) + 1e-6

        assert not torch.isnan(mu).all()
        assert not torch.isinf(mu).all()
        assert not torch.isinf(sig).all()
        assert not torch.isnan(sig).all()

        return torch.cat([mu, sig], dim=-1)

