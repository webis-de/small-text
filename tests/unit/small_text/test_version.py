import unittest

from packaging.version import Version
from small_text import get_version

class VersionTest(unittest.TestCase):

    def test_version_var(self):
        from small_text.version import __version__
        self.assertTrue(isinstance(__version__, str))

    def test_get_version(self):
        self.assertTrue(isinstance(get_version(), Version))
