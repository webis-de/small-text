import unittest


class VersionTest(unittest.TestCase):

    def test_version(self):
        from small_text.version import __version__

        self.assertTrue(isinstance(__version__, str))
