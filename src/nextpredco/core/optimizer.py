from abc import ABC, abstractmethod
from typing import override

from nextpredco.core.descriptors import ReadOnlyStr


class Optimizer(ABC):
    name = ReadOnlyStr()

    def __init__(self, name: str, nlp: str, opts: dict):
        self._name = name
        self._nlp = nlp
        self._opts = opts

    @abstractmethod
    def solve(self):
        pass


class IPOPT(Optimizer):
    def __init__(self, name: str, nlp: str, opts: dict):
        super().__init__(name, nlp, opts)

    @override
    def solve(self):
        pass


class PSO(Optimizer):
    def __init__(self, name: str, nlp: str, opts: dict):
        super().__init__(name, nlp, opts)

    @override
    def solve(self):
        pass
