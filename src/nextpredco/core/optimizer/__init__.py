from nextpredco.core.optimizer._ipopt import IPOPT as IPOPT
from nextpredco.core.settings import (
    IPOPTSettings as IPOPTSettings,
)
from nextpredco.core.settings import (
    OptimizerSettings as OptimizerSettings,
)

type Optimizer = IPOPT

__all__ = ['IPOPT', 'Optimizer', 'OptimizerFactory']


class OptimizerFactory:
    @staticmethod
    def create(settings: OptimizerSettings) -> Optimizer:
        if isinstance(settings, IPOPTSettings):
            return IPOPT(settings)

        raise NotImplementedError
