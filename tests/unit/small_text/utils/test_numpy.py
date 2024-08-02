import unittest

import numpy as np
from unittest.mock import patch

from small_text.utils.numpy import int_dtype


class NumpyUtilsTest(unittest.TestCase):

    def test_int_dtype(self):
        with patch('numpy.__version__', new='2.0.0'):
            self.assertEqual(np.uintp, int_dtype())
        with patch('numpy.__version__', new='2.0.1'):
            self.assertEqual(np.uintp, int_dtype())

    def test_int_dtype_legacy_numpy(self):
        with patch('numpy.__version__', new='1.25.0'):
            self.assertEqual(np.int32, int_dtype())
