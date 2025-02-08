from nextpredco.core._typing import Symbolic
from nextpredco.core.integrator._idas import IDAS as IDAS
from nextpredco.core.integrator._taylors import Taylor as Taylor
from nextpredco.core.settings import (
    IDASSettings,
    IntegratorSettings,
    TaylorSettings,
)

type Integrator = IDAS | Taylor


class IntegratorFactory:
    @staticmethod
    def create(
        settings: IntegratorSettings,
        equations: dict[str, Symbolic],
        h: float | None = None,
    ) -> Integrator | None:
        if isinstance(settings, IDASSettings):
            return IDAS(settings, equations, h)
        if isinstance(settings, TaylorSettings):
            return Taylor(settings, equations, h)
        return None


__all__ = ['IDAS', 'Integrator', 'IntegratorFactory', 'Taylor']
