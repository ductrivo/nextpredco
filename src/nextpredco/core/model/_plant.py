from nextpredco.core.model._model2 import ModelABC as ModelABC


class Plant(ModelABC):
    def __init__(self, settings=None, integrator_settings=None):
        super().__init__(settings, integrator_settings)
