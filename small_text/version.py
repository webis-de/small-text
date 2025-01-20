import tomli
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

    with open(pyproject_toml, 'rb') as f:
        pyproject_toml_data = tomli.load(f)
        return parse(pyproject_toml_data['project']['version'])


__version__ = str(get_version())
