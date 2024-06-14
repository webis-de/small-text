import unittest

from small_text.exceptions import MissingOptionalDependencyError
from small_text.query_strategies.strategies import RandomSampling
from small_text.query_strategies.subsampling import AnchorSubsampling


class AnchorSubsamplingTest(unittest.TestCase):

    def test_init(self):
        with self.assertRaisesRegex(MissingOptionalDependencyError,
                                    'The optional dependency \'hnswlib\''):
            AnchorSubsampling(RandomSampling(), 20)
