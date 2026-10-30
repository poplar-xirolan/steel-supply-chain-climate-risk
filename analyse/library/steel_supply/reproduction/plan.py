"""The source-only scientific chain and explicit segments of that chain."""
import fnmatch
import json

from ..pipeline_paths import bundle_root, canonical_selector


def build_plan(first=None, last=None):
    stages = json.loads((bundle_root() / 'config/stages.json').read_text(encoding='utf-8'))
    for stage in stages:
        for key in ['inputs', 'outputs']:
            stage[key] = [canonical_selector(p) for p in stage[key]]
    ids = [s['id'] for s in stages]
    start = ids.index(first) if first else 0
    stop = ids.index(last) + 1 if last else len(stages)
    if start >= stop:
        raise ValueError('The first stage must not follow the last stage')
    return stages[start:stop]


def external_inputs(stages):
    produced, external = [], []
    for stage in stages:
        for value in stage['inputs']:
            if not any(value == p or value.startswith(p.rstrip('/') + '/') or fnmatch.fnmatch(value, p) for p in produced):
                if value not in external:
                    external.append(value)
        produced.extend(stage['outputs'])
    return external
