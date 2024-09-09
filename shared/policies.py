from typing import Tuple

class SoftPolicy:
    def get_greedy_action(self, s) -> Tuple[int, float]:
        raise NotImplementedError

    def get_sa_probability(self, s, a) -> float:
        raise NotImplementedError