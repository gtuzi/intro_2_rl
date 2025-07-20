from typing import Union, List, Tuple, Optional
import torch
from sympy.physics.mechanics import Torque
from torch import nn

import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Categorical


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
                out_features=self.n_actions)
            )

        else:
            layers = [
                nn.Linear(self.in_size, self.n_actions),
            ]

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def logprob_s(self, x: Tensor, temperature: float = 1.):
        return F.log_softmax(self.forward(x)/temperature, dim=-1)

    def prob_s(self, x: Tensor, temperature: float = 1.):
        return F.softmax(self.forward(x) / temperature, dim=-1)

    def prob_sa(self, x: Tensor, a: Tensor, temperature: float = 1.):
        """
            a values $\in$ [0, num_actions)
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