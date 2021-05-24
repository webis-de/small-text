import json
from pathlib import Path

with open(Path(__file__).parent.joinpath('version.json')) as f:
    version = json.load(f)

version_str = '.'.join(map(str, [version['major'], version['minor'], version['micro']]))
if version['local'] != '':
    version_str += '+' + version['local']

__version__ = version_str
