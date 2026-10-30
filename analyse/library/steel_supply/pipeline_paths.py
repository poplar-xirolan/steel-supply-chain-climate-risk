"""Explicit input, work and result roots; no lookup of an author's checkout."""
from dataclasses import dataclass
import os
from pathlib import Path

def bundle_root():
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunPaths:
    data: Path
    work: Path
    result: Path

    @classmethod
    def under(cls, root):
        root = Path(root).resolve()
        return cls(root / 'data', root / 'work', root / 'result')


def workspace_root():
    return Path(os.environ.get('STEEL_ANALYSE_ROOT', bundle_root())).expanduser().resolve()


def canonical_selector(selector):
    selector = str(selector).replace('\\', '/')
    if not selector:
        return ''
    if Path(selector).is_absolute() or '..' in Path(selector).parts:
        raise ValueError(f'Path escapes the selected workspace: {selector}')
    if selector == 'input/climate_facilities' or selector.startswith('input/climate_facilities/'):
        selector = 'work/prepared_climate_facilities' + selector[len('input/climate_facilities'):]
    elif selector == 'input' or selector.startswith('input/'):
        selector = 'data/external' + selector[5:]
    elif selector == 'results' or selector.startswith('results/'):
        selector = 'result' + selector[7:]
    return selector


def science_path(root, *parts):
    root = Path(root).expanduser().resolve()
    selector = canonical_selector('/'.join(str(p).replace('\\', '/') for p in parts))
    result = (root / selector).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path escapes the selected workspace: {result}')
    return result


def data_path(*parts):
    return science_path(workspace_root(), *parts)


def relative_data_path(path):
    return Path(path).resolve().relative_to(workspace_root()).as_posix()
