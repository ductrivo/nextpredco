from abc import ABC, abstractmethod
from typing import override

from nextpredco.core.optimizer._optimizer import Optimizer
from nextpredco.core.settings import IPOPTSettings


class IPOPT(Optimizer):
    def __init__(self, settings: IPOPTSettings):
        super().__init__(settings)

    @override
    def solve(self):
        pass
