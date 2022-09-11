import unittest

from packaging import version
from unittest.mock import patch

from small_text.utils.annotations import (
    deprecated,
    experimental,
    DeprecationError,
    ExperimentalWarning
)


class DeprecationUtilsTest(unittest.TestCase):

    def test_deprecated_invalid_target(self):
        with self.assertRaisesRegex(ValueError,
                                    'The @deprecated decorator requires a function or class'):
            deprecated(42, deprecated_in='1.0.0')

        # This cannot be prevented with the current mechanism but does not raise an error as well
        deprecated(None, deprecated_in='1.0.0')

    def test_deprecated_invalid_usage(self):
        with self.assertRaisesRegex(ValueError,
                                    r'Keyword argument \'deprecated_in\' must be set.'):
            @deprecated()
            def func():
                pass
        with self.assertRaisesRegex(ValueError,
                                    r'Keyword argument \'deprecated_in\' must be set.'):
            @deprecated
            def another_func():
                pass

    @patch('small_text.utils.annotations.get_version')
    def test_func_failure_of_removal(self, get_version_mock):
        get_version_mock.return_value = version.parse('10.0.0')

        with self.assertRaisesRegex(DeprecationError,
                                    r'The function myfunc should have been removed '
                                    r'before version 10\.0\.0\.'):
            @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0')
            def myfunc(somearg, other=123):
                return somearg == 'foo' and other == 123

    @patch('small_text.utils.annotations.get_version')
    def test_class_failure_of_removal(self, get_version_mock):
        get_version_mock.return_value = version.parse('10.0.0')

        with self.assertRaisesRegex(DeprecationError,
                                    r'The class NewClass should have been removed '
                                    r'before version 10\.0\.0\.'):
            @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0')
            class NewClass(object):
                pass

    def test_deprecate_function(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0')
        def myfunc(somearg, other=123):
            return somearg == 'foo' and other == 123

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function myfunc has been deprecated in 1\.0\.0\.'):
            self.assertTrue(myfunc('foo'))

    def test_deprecate_function_with_removed_in(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0')
        def myfunc(somearg, other=123):
            return somearg == 'foo' and other == 123

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function myfunc has been deprecated in 1\.0\.0\ '
                                   r'and will be removed in 2\.0\.0\.'):
            self.assertTrue(myfunc('foo'))

    def test_deprecate_function_with_replacement(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', replacement='otherpkg.my.other_func()')
        def myfunc(somearg, other=123):
            return somearg == 'foo' and other == 123

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function myfunc has been deprecated in 1\.0\.0\. '
                                   r'Please use otherpkg\.my\.other_func\(\) instead\.'):
            self.assertTrue(myfunc('foo'))

    def test_deprecate_function_with_removed_in_and_replacement(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0',
                    replacement='otherpkg.my.other_func()')
        def myfunc(somearg, other=123):
            return somearg == 'foo' and other == 123

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function myfunc has been deprecated in 1\.0\.0\ '
                                   r'and will be removed in 2\.0\.0\. '
                                   r'Please use otherpkg\.my\.other_func\(\) instead\.'):
            self.assertTrue(myfunc('foo'))

    def test_deprecate_class(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0')
        class MyClass(object):
            pass

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The class MyClass has been deprecated in 1\.0\.0\.'):
            MyClass()

    def test_deprecate_class_with_removed_in(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0')
        class MyClass(object):
            pass

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The class MyClass has been deprecated in 1\.0\.0 '
                                   r'and will be removed in 2\.0\.0\.'):
            MyClass()

    def test_deprecate_class_with_replacement(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', replacement='otherpkg.my.other_func()')
        class MyClass(object):
            pass

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The class MyClass has been deprecated in 1\.0\.0\. '
                                   r'Please use otherpkg\.my\.other_func\(\) instead\.'):
            MyClass()

    def test_deprecate_class_with_removed_in_and_replacement(self):
        from small_text.utils.annotations import deprecated

        @deprecated(deprecated_in='1.0.0', to_be_removed_in='2.0.0',
                    replacement='otherpkg.my.other_func()')
        class MyClass(object):
            pass

        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The class MyClass has been deprecated in 1\.0\.0 '
                                   r'and will be removed in 2\.0\.0\. '
                                   r'Please use otherpkg\.my\.other_func\(\) instead\.'):
            MyClass()


class ExperimentalDecoratorTest(unittest.TestCase):

    def test_experimental_invalid_target(self):
        with self.assertRaisesRegex(ValueError,
                                    'The @experimental decorator requires a function or class'):
            experimental(42)

        # This cannot be prevented with the current mechanism but does not raise an error as well
        experimental(None)

    def test_experimental_class(self):
        @experimental()
        class MyClass(object):
            pass

        with self.assertWarnsRegex(ExperimentalWarning,
                                   r'The class MyClass is experimental '
                                   r'and maybe subject to change soon.'):
            MyClass()

        @experimental
        class MyOtherClass(object):
            pass

        with self.assertWarnsRegex(ExperimentalWarning,
                                   r'The class MyOtherClass is experimental '
                                   r'and maybe subject to change soon.'):
            MyOtherClass()

    def test_experimental_function(self):
        @experimental()
        def my_func():
            pass

        with self.assertWarnsRegex(ExperimentalWarning,
                                   r'The function my_func is experimental '
                                   r'and maybe subject to change soon.'):
            my_func()

        @experimental
        def my_other_func():
            pass

        with self.assertWarnsRegex(ExperimentalWarning,
                                   r'The function my_other_func is experimental '
                                   r'and maybe subject to change soon.'):
            my_other_func()
