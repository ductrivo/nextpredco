from typing import override

from nextpredco.core.optimizer._optimizer import OptimizerABC
from nextpredco.core.settings import IPOPTSettings


class IPOPT(OptimizerABC):
    def __init__(self, settings: IPOPTSettings):
        super().__init__(settings)

    @override
    def solve(self):
        pass
