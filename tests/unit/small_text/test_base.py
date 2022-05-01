import unittest

import small_text
from small_text.base import check_optional_dependency
from small_text.exceptions import MissingOptionalDependencyError


class OptionalDependenciesTest(unittest.TestCase):

    def test_check_optional_dependency(self):
        small_text.base.OPTIONAL_DEPENDENCIES['not_installed'] = 'module_name_for_not_installed'

        with self.assertRaisesRegex(MissingOptionalDependencyError,
                                    'The optional dependency \'not_installed\' is required') \
                as context:
            check_optional_dependency('not_installed')

        self.assertEqual(MissingOptionalDependencyError, context.expected)

    @staticmethod
    def test_check_optional_dependency_when_dependency_is_present():
        small_text.base.OPTIONAL_DEPENDENCIES['numpy'] = 'numpy'
        check_optional_dependency('numpy')

    def test_check_optional_dependency_when_dependency_is_not_registered(self):
        expected_msg = 'The given dependency \'not_registered_in_optional_dependencies\' is not'
        with self.assertRaisesRegex(ValueError, expected_msg):
            check_optional_dependency('not_registered_in_optional_dependencies')
