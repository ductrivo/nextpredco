from abc import ABC, abstractmethod
from typing import override

from nextpredco.core.settings.settings import IPOPTSettings


class Optimizer(ABC):
    @property
    def settings(self) -> IPOPTSettings:
        return self._settings

    def __init__(self, settings: IPOPTSettings):
        self._settings = settings

    @abstractmethod
    def solve(self):
        pass


class IPOPT(Optimizer):
    def __init__(self, settings: IPOPTSettings):
        super().__init__(settings)

    @override
    def solve(self):
        pass
