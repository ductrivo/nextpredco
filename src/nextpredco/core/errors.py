from numpy.typing import NDArray


class StepSizeInitializationError(Exception):
    def __init__(self):
        super().__init__('Step size h must be initialized.')


class ReadOnlyAttributeError(AttributeError):
    """Exception raised when trying to set a read-only attribute."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Attribute '{name}' is read-only.")


class NotAvailableAttributeError(AttributeError):
    """Exception raised when trying to access an attribute
    that is not available."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Attribute '{name}' is not available.")


class SystemVariableError(ValueError):
    def __init__(self, subset: str, superset: str) -> None:
        super().__init__(f'{subset} must be a subset of {superset} variables.')


class PrimaryDefsLengthError(ValueError):
    def __init__(self):
        super().__init__(
            'The length of _primary_defs must be equal to '
            'the length of _sys_defs',
        )


class EmptyArrayError(ValueError):
    def __init__(self):
        super().__init('Cannot set value for an empty array.')


class MemoryAddressError(AssertionError):
    def __init__(self, name1: str, name2: str) -> None:
        super().__init__(
            f'{name1} and {name2} do not point to the same memory address.',
        )


class DescriptorSetError(AssertionError):
    def __init__(self, name: str, rand_val: NDArray) -> None:
        super().__init__(
            f'Failed to set {rand_val.T} to {name}.',
        )


class InvalidK0ValueError(ValueError):
    def __init__(self):
        super().__init__('k0 must be greater than or equal to 0.')
