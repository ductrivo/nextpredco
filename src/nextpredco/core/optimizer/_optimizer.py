from abc import ABC, abstractmethod

from nextpredco.core.settings import OptimizerSettings


class OptimizerABC(ABC):
    @property
    def settings(self) -> OptimizerSettings:
        return self._settings

    def __init__(self, settings: OptimizerSettings):
        self._settings = settings

    @abstractmethod
    def solve(self):
        pass
