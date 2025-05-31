"""
A Markov reward process, or MRP, is a Markov decision process without actions.
Use MRPs when focusing on the prediction problem, in which there is
no need to distinguish the dynamics due to the environment from those due to the
agent.
"""

from typing import Tuple
import numpy as np
import tqdm


class MRP5:
    """
        Random walk MRP from example 6.2 in the book.
    """
    def __init__(self):
        self.n_states = 5 + 1 # 5 + terminal
        # rows: from
        # cols: to
        self.p = np.array(
            [
                [0., .5, 0., 0., 0., .5],   # from A
                [.5, 0., .5, 0., 0., 0.],   # from B
                [0., .5, 0., .5, 0., 0.],   # from C
                [0., 0., .5, 0., .5, 0.],   # from D
                [0., 0., 0., .5, .0, .5],   # from E
                [0., 0., 0., .0, .0, 1.],   # from Terminal
            ]
        )

        self.r = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        self.r[4, 5] = 1. # Only E --> T has non-zero reward
        self.current_state = 2
        self.t = 0


    def reset(self, reset_time = False, initial_state = None):
        if initial_state is None:
            self.current_state = 2
        else:
            self.current_state = initial_state

        if reset_time:
            self.t = 0
        return self.current_state

    def step(self) -> Tuple[float, int, bool]:
        p = self.p[self.current_state]
        rs = self.r[self.current_state]

        # Hop
        self.current_state: int = np.random.choice(
            list(range(self.n_states)), p=p)
        self.t += 1

        return rs[self.current_state], self.current_state, self.current_state == 5

    @staticmethod
    def estimate_state_values(num_experiments=10):
        """
            Estimate the return for each state if starting from that state
        """
        Tmax = 200
        n_states = 5
        rewards_over_initial_states = {}

        for sinit in tqdm.tqdm(range(n_states), desc='State Value Estimation'):
            rewards_over_initial_states[sinit] = []
            for experiment in range(num_experiments):
                env = MRP5()
                s = env.reset(initial_state=sinit)

                for t in range(Tmax):
                    r, s, terminal = env.step()
                    if terminal:
                        rewards_over_initial_states[sinit].append(r)
                        s = env.reset()

        return {
            s: np.mean(rewards_over_initial_states[s])
            for s in range(n_states)
        }


class MRPX:

    import tqdm

    """
        Random walk / MRP from example 7.1 in the book.
    """

    def __init__(self, n_states: int = 19):
        self.n_states = n_states + 1 # n_states + terminal
        # rows: from
        # cols: to
        self.p = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        self.r = np.zeros((self.n_states, self.n_states), dtype=np.float32)

        for s in range(self.n_states):
            if s == 0:
                # First non-terminal state (A), -1 reward on the left
                self.p[s, s + 1] = 0.5
                self.p[s, -1] = 0.5
                self.r[s, -1] = -1.
            elif s + 1 == self.n_states - 1:
                self.p[s, s - 1] = 0.5
                self.p[s, s + 1] = 0.5
                # Last non-terminal state, 1/0 ? reward on the right.
                # Not too clear on example 7.1
                self.r[s, s + 1] = 1.
            elif s + 1 == self.n_states:
                # Terminal state, r = 0
                self.p[s, s] = 1.0
            else:
                # Others, non-terminal, 0-rewards everywhere
                self.p[s, s - 1] = 0.5
                self.p[s, s + 1] = 0.5

        self.current_state = self.n_states // 2
        self.t = 0

    def reset(self, reset_time = False, initial_state = None):

        if initial_state is None:
            self.current_state = self.n_states // 2
        else:
            self.current_state = initial_state

        if reset_time:
            self.t = 0
        return self.current_state

    def step(self) -> Tuple[float, int, bool]:
        p = self.p[self.current_state]
        rs = self.r[self.current_state]

        # Hop
        self.current_state: int = np.random.choice(
            list(range(self.n_states)), p=p)
        self.t += 1

        return (
            rs[self.current_state],
            self.current_state,
            self.current_state + 1 == self.n_states  # done
        )

    @staticmethod
    def estimate_state_values(n_states = 19, num_experiments = 10):
        """
            Estimate the return for each state if starting from that state
        """
        Tmax = 200
        rewards_over_initial_states = {}

        for sinit in tqdm.tqdm(range(n_states), desc='State Value Estimation'):
            rewards_over_initial_states[sinit] = []
            for experiment in range(num_experiments):
                env = MRPX(n_states)
                s = env.reset(initial_state=sinit)

                for t in range(Tmax):
                    r, s, terminal = env.step()
                    if terminal:
                        rewards_over_initial_states[sinit].append(r)
                        s = env.reset()

        return {
            s: np.mean(rewards_over_initial_states[s])
            for s in range(n_states)
        }


if __name__ == '__main__':

    print("====== MRP5 ======")

    env = MRP5()
    s = env.reset()
    print(f't:{-1}, r={-1}, s={s}, d={-1}')

    for t in range(20):
        r, s, d = env.step()
        print(f't:{t}, r={r}, s={s}, d={d}')
        if d:
            print('resetting...')
            s = env.reset()
            print(f't:{-1}, r={-1}, s={s}, d={-1}')
    state_value_est = MRP5.estimate_state_values(num_experiments=50)
    print("====== MRP5 Expected Return / Initial State ======")
    _ = [print(f'{s}: {state_value_est[s]: .2f}') for s in range(5)]

    print("====== MRP19 ======")

    n_states = 19
    state_value_est = MRPX.estimate_state_values(
        n_states=n_states, num_experiments=50)

    print("====== MRP19 Expected Return / Initial State ======")
    _ = [print(f'{s}: {state_value_est[s]: .2f}') for s in range(n_states)]

    exit(0)
