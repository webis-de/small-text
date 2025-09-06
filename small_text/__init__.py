from small_text.version import __version__

# this is the only file in this project where star imports are allowed
from small_text import (
    classifiers,
    data,
    initialization,
    integrations,
    query_strategies,
    stopping_criteria,
    training,
    utils,
    vector_indexes
)

from small_text.classifiers import *
from small_text.data import *
from small_text.initialization import *
from small_text.integrations import *
from small_text.integrations import pytorch, transformers
from small_text.query_strategies import *
from small_text.stopping_criteria import *
from small_text.training import *
from small_text.utils import *
from small_text.vector_indexes import *

from small_text.active_learner import (
    ActiveLearner,
    AbstractPoolBasedActiveLearner,
    PoolBasedActiveLearner
)
from small_text.base import (
    LABEL_UNLABELED,
    LABEL_IGNORED,
    OPTIONAL_DEPENDENCIES,
    check_optional_dependency
)
from small_text.exceptions import (
    ActiveLearnerException,
    ConstraintViolationError,
    LearnerNotInitializedException,
    MissingOptionalDependencyError
)
from small_text.version import get_version
from small_text.utils.system import is_pytorch_available, is_transformers_available


if is_pytorch_available():
    from small_text.integrations.pytorch import *

if is_transformers_available():
    from small_text.integrations.transformers import *


__all__ = [
    'ActiveLearner',
    'AbstractPoolBasedActiveLearner',
    'PoolBasedActiveLearner',
    'LABEL_UNLABELED',
    'LABEL_IGNORED',
    'OPTIONAL_DEPENDENCIES',
    'check_optional_dependency',
    'ActiveLearnerException',
    'ConstraintViolationError',
    'LearnerNotInitializedException',
    'MissingOptionalDependencyError',
    'get_version'
]

__all__ += classifiers.__all__
__all__ += data.__all__
__all__ += initialization.__all__
__all__ += query_strategies.__all__
__all__ += stopping_criteria.__all__
__all__ += training.__all__
__all__ += utils.__all__
__all__ += vector_indexes.__all__

__all__ += pytorch.__all__
__all__ += transformers.__all__
