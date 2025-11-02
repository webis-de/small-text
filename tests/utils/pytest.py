import importlib
import pytest


def mark_optional_dependency_test(dependency_name):
    def decorator(func):
        func = pytest.mark.pytorch(func)
        func = pytest.mark.skipif(importlib.util.find_spec(dependency_name) is None,
                                  reason=f'preconditions for dependency test not met: {dependency_name} not found')(func)
        return func
    return decorator
