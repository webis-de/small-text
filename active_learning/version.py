import json
from pathlib import Path

with open(Path(__file__).parent.joinpath('version.json')) as f:
    version = json.load(f)

version_str = '.'.join(map(str, [version['major'], version['minor'], version['micro']]))
if version['pre_release'] != '':
    version_str += '.' + version['pre_release']

__version__ = version_str
