import importlib
import numpy as np

from small_text.exceptions import MissingOptionalDependencyError


LABEL_IGNORED = -np.inf
"""Value that is used to ignore an example when passing the labels to the active learner's
update() method. Do not use this for your dataset.
"""

LABEL_UNLABELED = -1
"""Value that is used to indicate an unlabeled example (in the single-class scenario).
This can be used to create a completely unlabeled dataset.
"""


# map from requirement specifier to name of the module that should be tested for importing
OPTIONAL_DEPENDENCIES = dict({
    'hnswlib': 'hnswlib',
    'setfit': 'setfit'
})


def check_optional_dependency(dependency_name):
    try:
        if dependency_name not in OPTIONAL_DEPENDENCIES.keys():
            raise ValueError(f'The given dependency \'{dependency_name}\' is not registered '
                             f'as an optional dependency.')

        importlib.import_module(OPTIONAL_DEPENDENCIES[dependency_name])
    except ImportError:
        exception_msg = f'The optional dependency \'{dependency_name}\' is required ' \
                        f'to use this functionality.'
        raise MissingOptionalDependencyError(exception_msg)
