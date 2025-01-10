import toml
from pathlib import Path

from packaging.version import parse


def get_version():
    """Returns the current version.

    Returns
    -------
    version : packaging.version.Version
        A version object.
    """
    main_package = Path(__file__).parent
    pyproject_toml = main_package.parent / 'pyproject.toml'

    pyproject_toml_data = toml.load(pyproject_toml)

    return parse(pyproject_toml_data['project']['version'])


__version__ = str(get_version())
