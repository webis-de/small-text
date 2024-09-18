import unittest
import pytest

from small_text.exceptions import MissingOptionalDependencyError
from small_text.query_strategies.strategies import RandomSampling, LeastConfidence
from small_text.query_strategies.subsampling import AnchorSubsampling, SEALS


@pytest.mark.optional
class AnchorSubsamplingTest(unittest.TestCase):

    def test_init(self):
        with self.assertRaisesRegex(MissingOptionalDependencyError,
                                    'The optional dependency \'hnswlib\''):
            AnchorSubsampling(RandomSampling(), 20)


@pytest.mark.optional
class SEALSTest(unittest.TestCase):

    def test_init(self):
        with self.assertRaisesRegex(MissingOptionalDependencyError,
                                    'The optional dependency \'hnswlib\''):
            SEALS(LeastConfidence())
