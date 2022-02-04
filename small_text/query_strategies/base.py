from enum import Enum
from functools import partial, wraps

from scipy.sparse import csr_matrix

# args: (self, clf, x, x_indices_unlabeled, x_indices_labeled, y)
NUM_QUERY_ARGS = 6


class ClassificationType(Enum):
    SINGLE_LABEL = 'single-label'
    MULTI_LABEL = 'multi-label'

    @staticmethod
    def from_str(classification_type_str):
        if classification_type_str == 'single-label':
            return ClassificationType.SINGLE_LABEL
        elif classification_type_str == 'multi-label':
            return ClassificationType.MULTI_LABEL
        else:
            raise ValueError('Cannot convert string to classification type enum: '
                             f'{classification_type_str}')


def constraints(cls=None, classification_type=None):
    """

    """
    if not callable(cls):
        return partial(constraints, classification_type=classification_type)

    @wraps(cls, updated=())
    class QueryStrategyConstraints(cls):

        def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, *args, n=10, **kwargs):

            if classification_type is not None:
                if isinstance(classification_type, str):
                    classification_type_ = ClassificationType.from_str(classification_type)

                if classification_type_ == ClassificationType.SINGLE_LABEL and isinstance(y, csr_matrix):
                    raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                       f'classification_type={str(classification_type_.value)} '
                                       f'but multi-label data was encountered')
                elif classification_type_ == ClassificationType.MULTI_LABEL \
                        and not isinstance(y, csr_matrix):
                    raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                       f'classification_type={str(classification_type_.value)} '
                                       f'but single-label data was encountered')

            return super().query(clf, x, x_indices_unlabeled, x_indices_labeled, y,
                                 *args, n=n, **kwargs)

    return QueryStrategyConstraints
