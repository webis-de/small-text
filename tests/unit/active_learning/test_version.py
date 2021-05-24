import unittest


class VersionTest(unittest.TestCase):

    def test_version(self):
        from active_learning.version import __version__

        self.assertTrue(isinstance(__version__, str))
