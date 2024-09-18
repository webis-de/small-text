import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--optional',
        action='store_true',
        default=False,
        help='Run tests that require optional dependencies'
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--optional'):
        skip = pytest.mark.skip(reason='Only run when --optional is passed')
        for item in items:
            if 'optional' in item.keywords:
                item.add_marker(skip)
