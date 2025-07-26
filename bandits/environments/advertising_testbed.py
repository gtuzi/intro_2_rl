import numpy as np
import matplotlib
from typing import List, Tuple
from matplotlib import pyplot as plt


def visualize_trajectories(m):
    """
        Ref: https://github.com/ikatsov/tensor-house/blob/master/promotions/next-best-action-rl.ipynb
    :param m: matrix of trajectories to plot
    :return:
    """
    max_val = np.max(m)
    colors = ['white', '#fde725', '#35b779']
    if max_val == 2:
        colors = ['white', '#fde725', '#35b779']
    if max_val == 3:
        colors = ['white', '#440154', '#21918c', '#fde725']

    cmap = matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 20))
    chart = ax.imshow(m, cmap=cmap, interpolation='none')
    ax.set_aspect(0.5 * m.shape[1] / m.shape[0])
    ax.set_xticklabels(list(range(m.shape[0])))
    ax.set_yticklabels(list(range(m.shape[1])))
    ax.grid(True)
    ax.set_xlabel('Time')
    ax.set_ylabel('User ID')
    plt.colorbar(chart, fraction=0.025, pad=0.04, ax=ax)
    plt.show()


class Actions:
    NO_ACTION = 0
    AD = 1
    SMALL_OFFER = 2
    LARGE_OFFER = 3


class User:
    # User behaviors
    NO_ACTION = 0
    VISIT = 1
    PURCHASE = 2

    # Demographics
    YOUNG = 0
    OLD = 1
    UNKNOWN = 2

    # No-action, Visit, Purchase

    PASSIVE = [0.90, 0.08, 0.02] # User most likely to do nothing
    # PASSIVE = [1.0, 0.0, 0.0] # Deterministic do nothing

    INTERESTED = [0.80, 0.15, 0.05] # Increased interest in visits, and slightly higher purchasing
    # INTERESTED = [0, 1.0, 0.0] # Deterministic visit

    EAGER = [0.70, 0.08, 0.22] # The most likely to purchase
    # EAGER = [0.0, 0.0, 1.0]  # Deterministic purchase

    def __init__(self, demographic: int, noise: float = 0.0):
        assert demographic in [User.YOUNG, User.OLD, User.UNKNOWN]
        self.demographic = demographic
        self.noise = noise
        self.reset()

    @property
    def state(self) -> np.ndarray:
        self.set_offers_received_sequence()
        return np.concatenate([self.positive_offers_received_sequence, [self.demographic]])

    def reset(self):
        self.all_offers_received = list()
        self.positive_offers_received_sequence: np.ndarray = np.array([])
        self._behavior_probabilities = User.PASSIVE

    def _parse_offer_seq(self, f: np.ndarray) -> np.ndarray:
        """ Get the sequence of last 3 positive (i.e. non-no_action) actions"""
        if len(f) > 0:
            return f[np.where(f > Actions.NO_ACTION)]
        else:
            return np.array([])

    def set_offers_received_sequence(self):
        self.positive_offers_received_sequence = self._parse_offer_seq(np.array(self.all_offers_received))

        # Get the sequence of the last, at-most 3, positive actions. Padd with no-actions
        K = 3 - self.positive_offers_received_sequence[-3:].shape[0]

        offers_received_sequence_ = np.concatenate(
            (self.positive_offers_received_sequence[-3:],
             Actions.NO_ACTION * np.ones(K))
        )

        assert len(offers_received_sequence_) == 3

        self.positive_offers_received_sequence = offers_received_sequence_

    def _update_young(self):
        """ Update behavior probabilities for young demographic """
        self.set_offers_received_sequence()

        if (  # Saw an ad --> large offer -> eager
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[1] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[1] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER)
        ):
            self._behavior_probabilities = User.EAGER
        elif (
                # Saw an ad --> small offer -> interested
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[1] == Actions.SMALL_OFFER) or
                (self.positive_offers_received_sequence[1] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.SMALL_OFFER) or
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.SMALL_OFFER)
        ):
            self._behavior_probabilities = User.INTERESTED
        else:
            # Passive user
            self._behavior_probabilities = User.PASSIVE  # default behavior

    def _update_old(self):
        """ Update behavior probabilities for old demographic """
        self.set_offers_received_sequence()
        if (  # Saw an ad --> large offer -> eager
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[1] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[1] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER)
        ):
            self._behavior_probabilities = User.EAGER
        else:
            # Passive user
            self._behavior_probabilities = User.PASSIVE  # default behavior

    def _update_unknown(self):
        """ Update behavior for unknown demographic """
        self.set_offers_received_sequence()
        if (  # Saw an ad --> large offer -> eager
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[1] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[1] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER) or
                (self.positive_offers_received_sequence[0] == Actions.AD and
                 self.positive_offers_received_sequence[2] == Actions.LARGE_OFFER)
        ):
            self._behavior_probabilities = User.INTERESTED
        else:
            # Passive user
            self._behavior_probabilities = User.PASSIVE  # default behavior

    def _update_behavior_state(self):
        if self.demographic == User.YOUNG:
            self._update_young()
        elif self.demographic == User.OLD:
            self._update_old()
        else:
            self._update_unknown()

    def make_offer(self, offer: int):
        """ Send out an offer (action), e.g. via e-mail """
        self.all_offers_received.append(offer)

    def step(self) -> int:
        """ The simulation simulation_step. Observe user behavior at each simulation simulation_step """
        self._update_behavior_state()
        p = self._behavior_probabilities

        # add some eagerness noise. You never know !!
        if np.random.binomial(1, self.noise) > 0:
            p = User.EAGER

        behavior = int(np.random.choice([User.NO_ACTION, User.VISIT, User.PURCHASE], p=p, size=(1,)))

        return behavior


class AdvertisingTestBed:
    """
        Based on "Building a Next Best Action model using reinforcement learning" blog:
        https://blog.griddynamics.com/building-a-next-best-action-model-using-reinforcement-learning/

        with implementation in:
        https://github.com/ikatsov/tensor-house/blob/master/promotions/next-best-action-rl.ipynb
    """

    def __init__(self, n_users, noise: float = 0.01):
        assert 0 <= noise <= 1.
        demographics = np.random.randint(User.YOUNG, User.UNKNOWN + 1, size=n_users).tolist()
        self.users = [User(demo, noise=noise) for demo in demographics]
        self.purchases = np.zeros(shape=(n_users,))
        self.behavior_trajectories = list()

    def get_state_rewards(self) -> Tuple[List[float], List[float]]:
        states = np.stack([u.state for u in self.users], axis=0)
        rewards = self.purchases
        return states.tolist(), rewards.tolist()

    def make_offers(self, offers: List[int]) -> Tuple[List[float], List[float]]:
        assert isinstance(offers, List)
        assert isinstance(offers[0], int)

        """ Take agent's offer, generate the state and the rewards. Reset rewards """
        _ = [u.make_offer(a) for u, a in zip(self.users, offers)]
        states, rewards = self.get_state_rewards()
        self.purchases = np.zeros_like(self.purchases)  # Purchases in one transition
        return states, rewards

    def simulation_step(self):
        behaviors = np.array([u.step() for u in self.users])
        self.purchases += (behaviors == User.PURCHASE).astype(int)
        self.behavior_trajectories.append(behaviors)


if __name__ == '__main__':
    offer_gen_fn = lambda n: np.random.choice(
        [Actions.NO_ACTION, Actions.AD, Actions.SMALL_OFFER, Actions.LARGE_OFFER],
        p=[0.1, 0.45, 0.00, 0.45],
        size=n).tolist()

    n_users = 200
    T_simu = 120
    noise = 0.0
    ads_offer = (Actions.AD * np.ones(n_users, dtype=int)).tolist()
    small_offer = (Actions.SMALL_OFFER * np.ones(n_users, dtype=int)).tolist()
    large_offer = (Actions.LARGE_OFFER * np.ones(n_users, dtype=int)).tolist()

    T_offers = [30, 60, 90]
    # offers = {T_offers[0]: ads_offer, T_offers[1]: small_offer, T_offers[2]: large_offer}
    offers = {t: offer_gen_fn(n_users) for t in T_offers}

    testbed = AdvertisingTestBed(n_users, noise=noise)
    # s[0], r[0], a[0]
    states, rewards = testbed.get_state_rewards()
    actions = [(Actions.NO_ACTION * np.ones((n_users,), dtype=float)).tolist()]

    for tsimu in range(T_simu):
        if tsimu in T_offers:
            s_next, r = testbed.make_offers(offers[tsimu])
            states.extend(s_next)
            rewards.extend(r)
            actions.append(offers[tsimu])
        else:
            testbed.simulation_step()

    visualize_trajectories(np.array(testbed.behavior_trajectories).T)
    visualize_trajectories(np.array(actions).T)

    exit(0)
