import nox


@nox.session
def unit_tests(session):
    requirements = nox.project.load_toml('pyproject.toml')['project']['dependencies']
    session.install(*requirements)
    session.install('-r', 'requirements-dev.txt')

    session.run('pytest', '-m', 'not pytorch and not optional', 'tests/unit')
