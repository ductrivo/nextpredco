from abc import ABC, abstractmethod


def get_nested_attr(obj, attr_keys):
    for key in attr_keys:
        obj = getattr(obj, key)
    return obj


class ElementABC(ABC):
    def __init__(self):
        self._inputs = {}
        self._outputs = {}

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @abstractmethod
    def make_step(self):
        pass


# if __name__ == '__main__':
#     ElementABC()
