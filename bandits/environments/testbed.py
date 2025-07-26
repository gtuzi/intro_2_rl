from abc import ABC, abstractmethod


class NonAssocativeTestBed(ABC):
    @property
    @abstractmethod
    def best_mean(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def best_arm(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def arm_mean(self, a) -> float:
        raise NotImplementedError
