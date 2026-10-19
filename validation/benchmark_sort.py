"""Measure CPU transparency ordering independently of OpenGL and readback.

Run from the extracted project with its dependencies installed. Optionally pass
--reference-source /path/to/previous/src/unit_cell_gui/_transparency_sort.py.
The reference module is executed as Python code and must be trusted.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import unit_cell_gui
from unit_cell_gui import _transparency_sort as sorter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import rotation


def primitives(count, sparse):
    result = []
    for index, poly in enumerate(adjacent_cubes(count).polyhedra):
        # A sparse screen-space grid, independent of the shared-boundary case.
        vertices = poly.vertices
        if sparse:
            centre = np.array([(index % 16)*8., (index // 16)*8., 0.])
            vertices = vertices - poly.center + centre
        result.extend((False, vertices[list(face)], (poly.site_key,)) for face in poly.faces)
        edges = sorted({tuple(sorted((face[i], face[(i+1) % len(face)])))
                        for face in poly.faces for i in range(len(face))})
        result.extend((True, vertices[list(edge)], (poly.site_key,)) for edge in edges)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path)
    parser.add_argument('--samples', type=int, default=30)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error('at least ten samples are required')
    implementations = {'current': sorter.back_to_front}
    reference_hash = None
    if args.reference_source:
        spec = importlib.util.spec_from_file_location('_sort_benchmark_reference', args.reference_source)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        implementations['reference'] = reference.back_to_front
        reference_hash = hashlib.sha256(args.reference_source.read_bytes()).hexdigest()
    data = {'version': unit_cell_gui.__version__, 'python': platform.python_version(),
            'numpy': np.__version__, 'metric': 'CPU back_to_front only; no Qt/GL/readback',
            'source_sha256': hashlib.sha256(Path(sorter.__file__).read_bytes()).hexdigest(),
            'reference_sha256': reference_hash, 'cases': {}}
    for count, sparse in ((2,False),(12,False),(48,False),(16,True),(64,True),(256,True)):
        items = primitives(count, sparse)
        orientation = rotation(0,23) @ rotation(1,37)
        name = f'{"sparse" if sparse else "shared"}_{count}'
        entry = {'polyhedra': count, 'primitives': len(items), 'measurements': {}}
        orders = {}
        inputs = {'current': sorter.prepare_primitives(items), 'reference': items}
        entry['current_world_preparation_excluded'] = True
        # Warm each implementation, then alternate measurement order to reduce
        # bias from one implementation always running earlier in the process.
        for key, function in implementations.items():
            orders[key] = function(inputs[key], orientation, line_margin=.002)
            entry['measurements'][key] = {'samples_ms': []}
        for sample in range(args.samples):
            keys = list(implementations)
            if sample % 2:
                keys.reverse()
            for key in keys:
                start = time.perf_counter()
                order = implementations[key](inputs[key], orientation, line_margin=.002)
                elapsed = (time.perf_counter()-start)*1000
                if order != orders[key]:
                    raise AssertionError('nondeterministic sort')
                entry['measurements'][key]['samples_ms'].append(elapsed)
        for measurement in entry['measurements'].values():
            measurement['p50_ms'], measurement['p95_ms'] = (
                float(value) for value in np.percentile(measurement['samples_ms'], [50,95]))
        if 'reference' in orders:
            entry['same_order_and_diagnostics'] = orders['current'] == orders['reference']
            if not entry['same_order_and_diagnostics']:
                raise AssertionError(f'order differs from reference: {name}')
        entry['conflicts'] = orders['current'][1]
        candidate_count = 0
        original = sorter._overlapping_pair_batches
        def counted(*positional, **keywords):
            nonlocal candidate_count
            for pairs in original(*positional, **keywords):
                candidate_count += len(pairs)
                yield pairs
        with patch.object(sorter, '_overlapping_pair_batches', counted):
            sorter.back_to_front(inputs['current'], orientation, line_margin=.002)
        entry['bounds_candidates'] = candidate_count
        data['cases'][name] = entry
        print(name, {key: round(value['p50_ms'],2)
                     for key,value in entry['measurements'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__ == '__main__':
    main()
