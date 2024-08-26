import unittest
from random import randint, random
from agents import PrioritizedQueue, DeterministicTabularModel
from tabular_methods.utils import Experience
from collections import defaultdict

class QueueTest(unittest.TestCase):

    def setUp(self):
        # Code to set up resources before each test
        self.queue = PrioritizedQueue()
        self.sa_list = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        self.priorities = [3.5, -2, 20., -5, .6]
        self.sorted_idx = sorted(
            range(len(self.sa_list)), key=lambda k: self.priorities[k],
            reverse=True)

        for sa, p in zip(self.sa_list, self.priorities):
            self.queue.insert(sa, p)

    def tearDown(self):
        # Code to clean up resources after each test
        pass

    def test_01_insert(self):
        print("test_01_insert() ... ")

        self.assertTrue(
            self.queue.priorities == [
                self.priorities[i] for i in self.sorted_idx])
        self.assertTrue(
            self.queue.state_actions == [
                self.sa_list[i] for i in self.sorted_idx])

    def test_02_pop_first(self):
        print("test_02_first() ... ")
        for i in self.sorted_idx:
            p, sa = self.queue.pop_first()
            self.assertEqual(p, self.priorities[i])
            self.assertEqual(sa, self.sa_list[i])

    def test_03_empty(self):
        print("test_03_empty() ... ")
        while len(self.queue.priorities) > 0:
            _, _ = self.queue.pop_first()

        self.assertTrue(self.queue.is_empty())


class DeterministicTabularModelTest(unittest.TestCase):
    def setUp(self):
        # Code to set up resources before each test
        action_space_dims = 5
        state_space_dims = 11
        num_samples = 10

        get_rand_action = lambda: randint(
            -action_space_dims // 2, action_space_dims // 2)
        get_rand_state = lambda: randint(
            -state_space_dims // 2, state_space_dims // 2)
        get_rand_reward = lambda: 3. * random()

        self.experiences = [
            Experience(
                s=get_rand_state(),
                a=get_rand_action(),
                r=get_rand_reward(),
                sp=get_rand_state(),
                done=0.)
            for _ in range(num_samples - 1)
        ]

        self.experiences.append(
            Experience(
                s=get_rand_state(),
                a=get_rand_action(),
                r=get_rand_reward(),
                sp=get_rand_state(),
                done=1.)
        )

        self.times = list(range(num_samples))
        self.model = DeterministicTabularModel(action_space_dims)
        self.model.initialize()

    def test_01_learn(self):
        print("test_01_learn() ... ")
        for e, t in zip(self.experiences, self.times):
            self.model.learn(e, t)
            # Deterministic model.
            # just keep track of the latest experience.
            self.assertEqual(
                self.model.get_nex_state_reward_terminal(e.s, e.a),
                (e.sp, e.r))

    def test_02_sa_list_that_lead_to_s(self):
        print("test_02_sa_list_that_lead_to_s() ...")

        # track the (s,a) that lead to sp
        inverse_transitions = defaultdict(lambda: [])
        for e, t in zip(self.experiences, self.times):
            self.model.learn(e, t)
            inverse_transitions[e.sp].append((e.s, e.a))

        for s, sa_bar in inverse_transitions.items():
            # The order of (s, a) -> sp doesn't matter
            self.assertEqual(
                set(sa_bar),
                set(self.model.get_sa_list_that_lead_to_s(s))
            )

    def tearDown(self):
        # Code to clean up resources after each test
        pass


if __name__ == '__main__':
    unittest.main()
