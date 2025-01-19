import nox


@nox.session
@nox.parametrize(
    'python,numpy_version',
    [
        (python, numpy)
        for python in ('3.9', '3.10', '3.11')
        for numpy in ('<2.0.0', '>=2.0.0')
    ],
)
def unit_tests(session, numpy_version):
    requirements = nox.project.load_toml('pyproject.toml')['project']['dependencies']
    session.install(*requirements)
    session.install('-r', 'requirements-dev.txt')

    session.install(f'numpy{numpy_version}')

    session.run('pytest', '-m', 'not pytorch and not optional', 'tests/unit')


@nox.session
@nox.parametrize(
    'python,numpy_version,torch_version',
    [
        (python, numpy, torch)
        for python in ('3.9', '3.10', '3.11')
        for numpy in ('<2.0.0', '>=2.0.0')
        for torch in ('==2.3.1', '==2.4.1', '==2.5.1')
    ],
)
def unit_tests_transformers(session, numpy_version, torch_version):
    toml_data = nox.project.load_toml('pyproject.toml')
    session.install(*toml_data['project']['dependencies'])
    session.install(*toml_data['project']['optional-dependencies']['transformers'])
    session.install('-r', 'requirements-dev.txt')

    session.install(f'numpy{numpy_version}')
    session.install(f'torch{torch_version}')

    # TODO: maybe also optional
    session.run('pytest', '-m', 'not optional', 'tests/unit')
