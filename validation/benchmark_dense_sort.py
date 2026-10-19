"""Measure candidates, graph constraints and traced peak storage on dense faces.

Parallel planes share projected bounds without physically intersecting. Graph
storage is intentionally unbounded here; the test documents its dense scaling.
tracemalloc runs separately from timings and covers traced allocations only.
"""
import argparse
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np
from unit_cell_gui._transparency_sort import prepare_primitives,back_to_front


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=[]
    for count in (32,64,128,256):
        square=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        geometry=prepare_primitives([(False,square+[0,0,i*.01],(str(i),)) for i in range(count)])
        samples=[];stats={}
        back_to_front(geometry,np.eye(3))
        for _ in range(10):
            start=time.perf_counter();order,conflicts=back_to_front(geometry,np.eye(3),stats=stats)
            samples.append((time.perf_counter()-start)*1000)
            assert order==list(range(count)) and conflicts==0
        tracemalloc.start();back_to_front(geometry,np.eye(3));_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        row={**stats,'samples_ms':samples,'p50_ms':float(np.median(samples)),
             'p95_ms':float(np.percentile(samples,95)),'separate_peak_traced_bytes':peak}
        results.append(row);print(count,stats['candidates'],stats['constraints'],peak,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scene':'parallel nonintersecting faces, common projected overlap',
                                      'memory':'separate tracemalloc peak; graph_bytes_estimate is an estimate, not a cap',
                                      'cases':results},indent=2)+'\n')


if __name__=='__main__':main()
